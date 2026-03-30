from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch

from dataset.utils import prepare_dataset
from text_encoder.dataset import TextGuidedYOLODataset
from text_encoder.embedding_store import ObjectTextEmbeddingStore
from text_encoder.matching import assign_by_score_matrix, box_iou_matrix, score_matrix_from_phrase_maps
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, TQDM
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.files import increment_path
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import non_max_suppression, scale_boxes, xywh2xyxy
from ultralytics.utils.torch_utils import select_device


PR_IOU_THRESHOLDS: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Text-guided validation + visualization with image embeddings and GT/Pred matching.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--weights", type=str,  default="runs/train/DIOR-RSVG/weights/best.pt", help="Path to trained best.pt checkpoint.")
	parser.add_argument("--data", type=str, default="", help="Optional existing data.yaml path. If set, skip dataset preparation.")
	parser.add_argument("--voc-root", type=str, default="/data/bxc/DIOR-RSVG", help="VOC root with images, annotations and split txt files.")
	parser.add_argument("--images-dir", type=str, default="JPGEImages", help="Image folder name under VOC root.")
	parser.add_argument("--annotations-dir", type=str, default="Annotations", help="XML annotation folder name.")
	parser.add_argument("--train-list", type=str, default="train.txt", help="Train split txt file name.")
	parser.add_argument("--val-list", type=str, default="val.txt", help="Val split txt file name.")
	parser.add_argument("--test-list", type=str, default="test.txt", help="Test split txt file name.")
	parser.add_argument("--prepared-dir", "--prepared_dir", dest="prepared_dir", type=str, default="pre_datasets/DIOR_RSVG", help="Output dir for generated data.yaml and mapping files.")
	parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to validate.")
	parser.add_argument("--embedding-dir",type=str,default="",help="Embedding directory containing *_text_embeddings.pt. Empty means parent directory of data.yaml.",)
	parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
	parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
	parser.add_argument("--workers", type=int, default=0, help="Dataloader workers.")
	parser.add_argument("--device", type=str, default="1", help="Device, e.g. '0', 'cpu'. Empty means auto.")
	parser.add_argument("--half", action="store_true", help="Use FP16 if device supports it.")
	parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for NMS.")
	parser.add_argument("--nms-iou", type=float, default=0.7, help="IoU threshold for NMS.")
	parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for TP/FP matching.")
	parser.add_argument("--max-det", type=int, default=300, help="Max detections per image.")
	parser.add_argument("--matching-temperature", type=float, default=0.7, help="Temperature for score-matrix Hungarian matching.")
	parser.add_argument("--max-images", type=int, default=0, help="Max images to process. 0 means all.")
	parser.add_argument("--project", type=str, default="runs/test", help="Project directory.")
	parser.add_argument("--name", type=str, default="DIOR", help="Experiment name.")
	parser.add_argument("--exist-ok", action="store_true", help="Reuse existing experiment directory.")
	parser.add_argument("--text-embedding-dim",type=int,default=0,help="Expected embedding dim. 0 means infer from model (fallback 768).",)
	parser.add_argument("--text-model-name", type=str, default="ViT-L-14", help="Text encoder model name for missing split embedding generation.")
	parser.add_argument("--text-pretrained", type=str, default="openai", help="Text encoder pretrained tag for missing split embedding generation.")
	parser.add_argument("--text-precision", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="Text encoding precision for missing split embedding generation.")
	parser.add_argument("--text-embed-batch-size", type=int, default=256, help="Batch size when generating missing split embeddings.")
	parser.add_argument("--text-overwrite", action="store_true", help="Overwrite selected split embedding when auto-generating.")
	parser.add_argument("--no-auto-build-split-embedding", action="store_true", help="Disable auto-generation for missing selected split embedding.")
	return parser.parse_args()


def _resolve_data_yaml(args: argparse.Namespace) -> Path:
	raw_data = str(args.data or "").strip()
	if raw_data:
		data_yaml = Path(raw_data).expanduser().resolve()
		if not data_yaml.exists():
			raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
		return data_yaml

	build_device = str(args.device or "").strip() or "auto"
	data_yaml = Path(prepare_dataset(args, device=build_device)).expanduser().resolve()
	if not data_yaml.exists():
		raise FileNotFoundError(f"Dataset preparation failed, missing data.yaml: {data_yaml}")
	return data_yaml


def _resolve_prepared_root(data: Dict[str, Any], data_yaml: Path) -> Path:
	root = str(data.get("path", "") or "").strip()
	if root:
		return Path(root).expanduser().resolve()
	return data_yaml.parent.resolve()


def _ensure_split_embedding(
	prepared_root: Path,
	embedding_root: Path,
	split: str,
	args: argparse.Namespace,
	text_device: str,
) -> Path:
	emb_file = embedding_root / f"{split}_text_embeddings.pt"
	if emb_file.exists() and not bool(args.text_overwrite):
		return emb_file

	embedding_root.mkdir(parents=True, exist_ok=True)
	from dataset.precompute_text_embeddings import build_openclip_text_embeddings

	LOGGER.info("Building text embeddings for split '%s' into %s", split, embedding_root)
	result = build_openclip_text_embeddings(
		prepared_dir=str(prepared_root),
		splits=[split],
		output_dir=str(embedding_root),
		model_name=str(args.text_model_name),
		pretrained=str(args.text_pretrained),
		device=text_device,
		precision=str(args.text_precision),
		batch_size=int(args.text_embed_batch_size),
		overwrite=bool(args.text_overwrite),
		runtime_meta={"caller": "test.py", "requested_split": split},
	)

	split_stats = (result.get("splits", {}) or {}).get(split, {})
	if str(split_stats.get("status", "")) == "empty_split":
		raise RuntimeError(f"Split '{split}' has no samples, cannot build text embeddings.")

	if not emb_file.exists():
		raise FileNotFoundError(
			f"Missing embedding file for split '{split}': {emb_file}. "
			"Check samples/<split>_samples.jsonl and text generation settings."
		)

	return emb_file


def _candidate_ids(sample_id: str) -> List[str]:
	"""Generate fallback keys for sample_id lookup, mirroring embedding store behavior."""
	raw = str(sample_id or "").strip()
	if not raw:
		return []

	candidates: List[str] = []

	def _push(value: str) -> None:
		value = str(value).strip()
		if value and value not in candidates:
			candidates.append(value)

	_push(raw)
	_push(raw.replace("\\", "/"))

	raw_path = Path(raw)
	_push(raw_path.name)
	_push(raw_path.stem)

	norm_path = Path(raw.replace("\\", "/"))
	_push(norm_path.name)
	_push(norm_path.stem)

	for key in list(candidates):
		if "__" in key:
			_push(key.split("__", 1)[1])

	return candidates


def _safe_text(value: Any) -> str:
	if value is None:
		return ""
	if isinstance(value, str):
		return value.strip()
	if isinstance(value, (list, tuple)):
		parts = [str(x).strip() for x in value if str(x).strip()]
		return " | ".join(parts)
	return str(value).strip()


def _batch_item(batch: Dict[str, Any], key: str, index: int) -> Any:
	value = batch.get(key)
	if isinstance(value, (list, tuple)) and 0 <= index < len(value):
		return value[index]
	return None


def _extract_sample_ids(batch: Dict[str, Any]) -> List[str]:
	bsz = int(batch["img"].shape[0])

	raw_ids = batch.get("sample_id")
	if isinstance(raw_ids, (list, tuple)) and len(raw_ids) == bsz:
		return [str(x) for x in raw_ids]

	raw_labels = batch.get("label_file")
	if isinstance(raw_labels, (list, tuple)) and len(raw_labels) == bsz:
		return [Path(str(x)).stem for x in raw_labels]

	raw_images = batch.get("im_file")
	if isinstance(raw_images, (list, tuple)) and len(raw_images) == bsz:
		return [Path(str(x)).stem for x in raw_images]

	return ["" for _ in range(bsz)]


def _load_text_lookup(embedding_root: Path) -> Dict[str, List[str]]:
	"""Load optional text strings from embedding payload for visualization headers."""
	files = sorted(embedding_root.glob("*_text_embeddings.pt"))
	if not files and (embedding_root / "embeddings").exists():
		files = sorted((embedding_root / "embeddings").rglob("*_text_embeddings.pt"))

	text_map: Dict[str, List[str]] = {}
	for emb_file in files:
		try:
			payload = torch.load(emb_file, map_location="cpu")
		except Exception:
			continue

		ids = payload.get("ids", [])
		source_texts = payload.get("source_texts", payload.get("texts", []))
		phrases = payload.get("phrases", [])

		texts = source_texts
		if not isinstance(texts, (list, tuple)) or len(texts) != len(ids):
			texts = phrases
		if not isinstance(ids, (list, tuple)) or not isinstance(texts, (list, tuple)) or len(ids) != len(texts):
			continue

		for sid, text_obj in zip(ids, texts):
			sid = str(sid)
			if isinstance(text_obj, str):
				items = [text_obj.strip()] if text_obj.strip() else []
			elif isinstance(text_obj, (list, tuple)):
				items = [str(x).strip() for x in text_obj if str(x).strip()]
			else:
				items = []

			if not items:
				continue

			for key in _candidate_ids(sid):
				text_map.setdefault(key, items)

	return text_map


def _lookup_text(text_map: Dict[str, List[str]], sample_id: str) -> List[str]:
	for key in _candidate_ids(sample_id):
		found = text_map.get(key)
		if found:
			return found
	return []


def _resolve_description(batch: Dict[str, Any], index: int, sample_id: str, text_map: Dict[str, List[str]]) -> str:
	description_text = _safe_text(_batch_item(batch, "description_text", index))
	if description_text:
		return description_text

	descriptions = _batch_item(batch, "descriptions", index)
	description_text = _safe_text(descriptions)
	if description_text:
		return description_text

	single_description = _safe_text(_batch_item(batch, "description", index))
	if single_description:
		return single_description

	emb_text = _lookup_text(text_map, sample_id)
	if emb_text:
		return " | ".join(emb_text)

	return ""


def _resolve_description_list(batch: Dict[str, Any], index: int, sample_id: str, text_map: Dict[str, List[str]]) -> List[str]:
	descriptions = _batch_item(batch, "descriptions", index)
	if isinstance(descriptions, (list, tuple)):
		items = [str(x).strip() for x in descriptions if str(x).strip()]
		if items:
			return items

	single = _resolve_description(batch, index, sample_id, text_map)
	if single:
		return [single]

	return []


def _prepare_backend_img(backend: AutoBackend, img: torch.Tensor) -> torch.Tensor:
	if getattr(backend, "fp16", False) and img.dtype != torch.float16:
		img = img.half()
	if getattr(backend, "nhwc", False):
		img = img.permute(0, 2, 3, 1)
	return img


def _run_text_inference(backend: AutoBackend, batch: Dict[str, Any]):
	txt_vec = batch.get("txt_vec")
	txt_token_mask = batch.get("text_token_mask")
	txt_phrase_weight = batch.get("text_phrase_weight")
	if not isinstance(txt_vec, torch.Tensor):
		raise RuntimeError("batch['txt_vec'] is missing; cannot run text-guided inference.")

	infer_model: Any = backend
	img = batch["img"]

	if hasattr(backend, "model") and hasattr(backend, "pt"):
		if not (getattr(backend, "pt", False) or getattr(backend, "nn_module", False)):
			raise RuntimeError("Text-guided validation requires a PyTorch .pt/.pth backend.")
		img = _prepare_backend_img(backend, img)
		infer_model = backend.model

	try:
		out = infer_model(
			img,
			augment=False,
			txt_vec=txt_vec,
			txt_token_mask=txt_token_mask,
			txt_phrase_weight=txt_phrase_weight,
		)
		phrase_logits = []
		last_outputs = getattr(infer_model, "_last_text_outputs", None)
		if isinstance(last_outputs, dict) and isinstance(last_outputs.get("phrase_logits"), list):
			phrase_logits = last_outputs.get("phrase_logits")
		return out, phrase_logits
	except TypeError as e:
		retry_on_wrapped_model = (
			infer_model is backend
			and ("txt_token_mask" in str(e) or "txt_phrase_weight" in str(e))
			and hasattr(backend, "model")
			and callable(getattr(backend, "model"))
			and (getattr(backend, "pt", False) or getattr(backend, "nn_module", False))
		)
		if retry_on_wrapped_model:
			retry_img = _prepare_backend_img(backend, batch["img"])
			out = backend.model(
				retry_img,
				augment=False,
				txt_vec=txt_vec,
				txt_token_mask=txt_token_mask,
				txt_phrase_weight=txt_phrase_weight,
			)
			phrase_logits = []
			last_outputs = getattr(backend.model, "_last_text_outputs", None)
			if isinstance(last_outputs, dict) and isinstance(last_outputs.get("phrase_logits"), list):
				phrase_logits = last_outputs.get("phrase_logits")
			return out, phrase_logits
		raise


def _embedding_dim_from_model(backend: AutoBackend, fallback: int = 768) -> int:
	model = getattr(backend, "model", None)
	if model is None:
		return fallback
	for attr in ("text_embed_dim", "txt_embed_dim", "embed_dim"):
		value = getattr(model, attr, None)
		try:
			value = int(value)
		except Exception:
			continue
		if value > 0:
			return value
	return fallback


def _preprocess_batch(batch: Dict[str, Any], model: AutoBackend) -> Dict[str, Any]:
	batch["img"] = batch["img"].to(model.device, non_blocking=True)
	batch["img"] = (batch["img"].half() if model.fp16 else batch["img"].float()) / 255

	for key in ("batch_idx", "cls", "bboxes"):
		value = batch.get(key)
		if isinstance(value, torch.Tensor):
			batch[key] = value.to(model.device)
	return batch


def _xywhn_to_native_xyxy(batch: Dict[str, Any], image_index: int) -> torch.Tensor:
	idx = batch["batch_idx"] == image_index
	bboxes = batch["bboxes"][idx]
	if int(bboxes.shape[0]) == 0:
		return bboxes.new_zeros((0, 4))

	bboxes = bboxes.clone()
	imgsz = batch["img"].shape[2:]
	gain = torch.tensor((imgsz[1], imgsz[0], imgsz[1], imgsz[0]), device=bboxes.device, dtype=bboxes.dtype)
	bboxes = xywh2xyxy(bboxes) * gain
	scale_boxes(imgsz, bboxes, batch["ori_shape"][image_index], ratio_pad=batch["ratio_pad"][image_index])
	return bboxes


def _match_predictions_one_to_one(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, iou_thr: float) -> Tuple[List[bool], List[float], List[int]]:
	n_pred = int(pred_xyxy.shape[0])
	if n_pred == 0:
		return [], [], list(range(int(gt_xyxy.shape[0])))
	if int(gt_xyxy.shape[0]) == 0:
		return [False] * n_pred, [0.0] * n_pred, []

	ious = box_iou_matrix(pred_xyxy, gt_xyxy)
	tp_flags: List[bool] = [False] * n_pred
	best_ious: List[float] = [0.0] * n_pred
	matched_gt: set[int] = set()

	row_ind, col_ind = assign_by_score_matrix(ious)
	for r, c in zip(row_ind.tolist(), col_ind.tolist()):
		if r < 0 or r >= n_pred:
			continue
		iou_value = float(ious[r, c].item())
		best_ious[r] = iou_value
		if iou_value >= float(iou_thr):
			tp_flags[r] = True
			matched_gt.add(int(c))

	for pi in range(n_pred):
		if best_ious[pi] > 0:
			continue
		best_ious[pi] = float(ious[pi].max().item()) if int(ious.shape[1]) else 0.0

	unmatched_gt = [gi for gi in range(int(gt_xyxy.shape[0])) if gi not in matched_gt]
	return tp_flags, best_ious, unmatched_gt


def _draw_label_block(
	image: np.ndarray,
	lines: List[str],
	anchor_xy: Tuple[int, int],
	color: Tuple[int, int, int],
	anchor: str = "tl",
) -> None:
	if not lines:
		return

	h, w = image.shape[:2]
	font = cv2.FONT_HERSHEY_SIMPLEX
	scale = 0.45
	thickness = 1
	line_h = 18
	text_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
	block_w = max((tw for tw, _ in text_sizes), default=0) + 10
	block_h = max(1, len(lines)) * line_h + 6

	ax, ay = int(anchor_xy[0]), int(anchor_xy[1])
	if anchor == "br":
		x1 = ax - block_w
		y1 = ay - block_h
	else:
		x1 = ax
		y1 = ay

	x1 = max(0, min(x1, w - block_w - 1))
	y1 = max(0, min(y1, h - block_h - 1))
	x2 = min(w - 1, x1 + block_w)
	y2 = min(h - 1, y1 + block_h)

	cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
	for li, line in enumerate(lines):
		y = y1 + 16 + li * line_h
		cv2.putText(image, line, (x1 + 4, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_box(
	image: np.ndarray,
	box_xyxy: Sequence[float],
	color: Tuple[int, int, int],
	label_lines: List[str],
	label_anchor: str = "tl",
) -> None:
	h, w = image.shape[:2]
	x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]
	x1 = max(0, min(x1, w - 1))
	y1 = max(0, min(y1, h - 1))
	x2 = max(0, min(x2, w - 1))
	y2 = max(0, min(y2, h - 1))
	cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

	if label_lines:
		if label_anchor == "br":
			anchor_xy = (x2 - 2, y2 - 2)
		else:
			anchor_xy = (x1 + 2, max(0, y1 - 4 - (18 * len(label_lines) + 6)))
		_draw_label_block(image, label_lines, anchor_xy, color, anchor="br" if label_anchor == "br" else "tl")


def _header_canvas(image: np.ndarray, lines: List[str]) -> np.ndarray:
	line_h = 22
	pad = 10
	header_h = max(48, pad * 2 + line_h * len(lines))
	canvas = np.full((header_h + image.shape[0], image.shape[1], 3), 16, dtype=np.uint8)
	canvas[:header_h] = (24, 24, 24)
	canvas[header_h:] = image

	y = pad + 15
	for line in lines:
		cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (238, 238, 238), 1, cv2.LINE_AA)
		y += line_h
	return canvas


def _wrap_description(text: str, image_width: int) -> List[str]:
	if not text:
		return ["text: <empty>"]
	approx_char_width = 9
	max_chars = max(20, (image_width - 24) // approx_char_width)
	wrapped = textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False)
	if not wrapped:
		return ["text: <empty>"]
	return [f"text: {wrapped[0]}"] + [f"      {line}" for line in wrapped[1:]]


def _sanitize_name(text: str) -> str:
	text = str(text or "").strip().replace("\\", "_").replace("/", "_")
	return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)[:64] or "sample"


def _resolve_run_dirs(args: argparse.Namespace) -> Tuple[Path, Path]:
	run_dir = increment_path(Path(args.project).expanduser() / str(args.name), exist_ok=bool(args.exist_ok), mkdir=True)
	vsi_dir = run_dir / "vsi_result"
	vsi_dir.mkdir(parents=True, exist_ok=True)
	return run_dir.resolve(), vsi_dir.resolve()


def _compute_pr_curve(conf_tp_pairs: List[Tuple[float, int]], total_gt: int) -> List[Tuple[float, float]]:
	if not conf_tp_pairs or total_gt <= 0:
		return []

	sorted_pairs = sorted(conf_tp_pairs, key=lambda x: x[0], reverse=True)
	tp_cum = 0
	fp_cum = 0
	points: List[Tuple[float, float]] = []
	for _, is_tp in sorted_pairs:
		if int(is_tp) == 1:
			tp_cum += 1
		else:
			fp_cum += 1
		precision = tp_cum / max(tp_cum + fp_cum, 1)
		recall = tp_cum / max(total_gt, 1)
		points.append((recall, precision))
	return points


def _plot_metrics_overview(metrics: Dict[str, float], save_path: Path) -> None:
	width, height = 980, 520
	canvas = np.full((height, width, 3), 248, dtype=np.uint8)
	cv2.rectangle(canvas, (0, 0), (width - 1, 80), (31, 45, 61), -1)
	cv2.putText(canvas, "Text-Guided Test Metrics", (24, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)

	labels = ["Accuracy", "Precision", "Recall", "Mean TP IoU"]
	values = [
		float(metrics.get("acc", 0.0)),
		float(metrics.get("precision", 0.0)),
		float(metrics.get("recall", 0.0)),
		float(metrics.get("mean_tp_iou", 0.0)),
	]
	colors = [(76, 101, 216), (35, 137, 218), (67, 160, 71), (255, 153, 51)]

	left, right = 70, width - 60
	top, bottom = 150, 430
	bar_area_w = right - left
	bar_w = int(bar_area_w / max(len(values) * 2, 1))
	gap = bar_w

	for idx, (name, value, color) in enumerate(zip(labels, values, colors)):
		x1 = left + idx * (bar_w + gap) + 60
		x2 = x1 + bar_w
		bar_h = int(max(0.0, min(1.0, value)) * (bottom - top))
		y1 = bottom - bar_h
		cv2.rectangle(canvas, (x1, y1), (x2, bottom), color, -1)
		cv2.rectangle(canvas, (x1, top), (x2, bottom), (90, 90, 90), 2)
		cv2.putText(canvas, f"{value:.4f}", (x1 - 4, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 40, 40), 1, cv2.LINE_AA)
		cv2.putText(canvas, name, (x1 - 10, bottom + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1, cv2.LINE_AA)

	meta_texts = [
		f"images={int(metrics.get('images', 0))}",
		f"gt={int(metrics.get('gt', 0))}",
		f"pred={int(metrics.get('pred', 0))}",
		f"tp={int(metrics.get('tp', 0))}",
		f"fp={int(metrics.get('fp', 0))}",
		f"fn={int(metrics.get('fn', 0))}",
	]
	meta_line = " | ".join(meta_texts)
	cv2.putText(canvas, meta_line, (24, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (50, 50, 50), 1, cv2.LINE_AA)
	cv2.imwrite(str(save_path), canvas)


def _plot_pr_curve(points: List[Tuple[float, float]], save_path: Path) -> None:
	width, height = 980, 620
	canvas = np.full((height, width, 3), 255, dtype=np.uint8)
	margin_left, margin_right, margin_top, margin_bottom = 90, 50, 70, 80
	plot_w = width - margin_left - margin_right
	plot_h = height - margin_top - margin_bottom

	cv2.putText(canvas, "PR Curve (class-agnostic)", (margin_left, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2, cv2.LINE_AA)
	cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (60, 60, 60), 2)

	for tick in range(0, 11):
		x = margin_left + int(plot_w * (tick / 10.0))
		y = margin_top + int(plot_h * (tick / 10.0))
		cv2.line(canvas, (x, margin_top), (x, margin_top + plot_h), (232, 232, 232), 1)
		cv2.line(canvas, (margin_left, y), (margin_left + plot_w, y), (232, 232, 232), 1)
		cv2.putText(canvas, f"{tick/10:.1f}", (x - 14, margin_top + plot_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
		cv2.putText(canvas, f"{1 - tick/10:.1f}", (36, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

	if points:
		poly = []
		for recall, precision in points:
			recall = float(max(0.0, min(1.0, recall)))
			precision = float(max(0.0, min(1.0, precision)))
			x = margin_left + int(recall * plot_w)
			y = margin_top + int((1.0 - precision) * plot_h)
			poly.append((x, y))
		if len(poly) > 1:
			cv2.polylines(canvas, [np.array(poly, dtype=np.int32)], False, (27, 94, 32), 3, cv2.LINE_AA)
		for x, y in poly[:: max(1, len(poly) // 25)]:
			cv2.circle(canvas, (x, y), 2, (27, 94, 32), -1)
	else:
		cv2.putText(canvas, "No valid PR points (empty predictions or GT).", (margin_left, margin_top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (90, 90, 90), 1, cv2.LINE_AA)

	cv2.putText(canvas, "Recall", (margin_left + plot_w // 2 - 25, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (55, 55, 55), 1, cv2.LINE_AA)
	cv2.putText(canvas, "Precision", (20, margin_top + plot_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (55, 55, 55), 1, cv2.LINE_AA)
	cv2.imwrite(str(save_path), canvas)


def _write_artifacts(
	run_dir: Path,
	vsi_dir: Path,
	stats: Dict[str, Any],
	weights: Path,
	data_yaml: Path,
	embedding_root: Path,
	args: argparse.Namespace,
	failed_lines: List[str],
	correct_lines: List[str],
	conf_tp_pairs: List[Tuple[float, int]],
) -> Dict[str, Path]:
	acc = float(stats["tp"] / max(stats["tp"] + stats["fp"] + stats["fn"], 1))
	precision = float(stats["tp"] / max(stats["pred"], 1))
	recall = float(stats["tp"] / max(stats["gt"], 1))
	mean_tp_iou = float(stats["tp_iou_sum"] / max(stats["tp_iou_count"], 1))
	m_iou = float(stats["match_iou_sum"] / max(stats["match_iou_count"], 1))
	c_iou = float(stats["match_ciou_sum"] / max(stats["match_ciou_count"], 1))
	pr_at_metrics = {
		f"pr_at_{thr:.1f}": float(stats[f"pr_at_{thr:.1f}_count"] / max(stats["pred"], 1))
		for thr in PR_IOU_THRESHOLDS
	}

	metrics = {
		"weights": str(weights),
		"data": str(data_yaml),
		"split": str(args.split),
		"embedding_root": str(embedding_root),
		"run_dir": str(run_dir),
		"vsi_dir": str(vsi_dir),
		"images": int(stats["images"]),
		"gt": int(stats["gt"]),
		"pred": int(stats["pred"]),
		"tp": int(stats["tp"]),
		"fp": int(stats["fp"]),
		"fn": int(stats["fn"]),
		"missing_embed_images": int(stats["missing_embed_images"]),
		"acc": acc,
		"precision": precision,
		"recall": recall,
		"mean_tp_iou": mean_tp_iou,
		"mIoU": m_iou,
		"cIoU": c_iou,
		**pr_at_metrics,
	}

	summary_json = run_dir / "metrics_summary.json"
	summary_txt = run_dir / "metrics_summary.txt"
	failed_txt = run_dir / "failed_objects.txt"
	correct_txt = run_dir / "correct_objects.txt"
	metrics_vis = run_dir / "metrics_overview.jpg"
	pr_curve_vis = run_dir / "pr_curve.jpg"
	pr_points_csv = run_dir / "pr_points.csv"

	summary_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
	summary_txt.write_text(
		"\n".join(
			[
				"=== Text-Guided Validation Summary ===",
				f"weights: {weights}",
				f"data: {data_yaml}",
				f"split: {args.split}",
				f"embedding root: {embedding_root}",
				f"run dir: {run_dir}",
				f"vsi result dir: {vsi_dir}",
				f"images: {metrics['images']}",
				f"total gt: {metrics['gt']}",
				f"total pred: {metrics['pred']}",
				f"tp: {metrics['tp']} | fp: {metrics['fp']} | fn: {metrics['fn']}",
				f"acc: {metrics['acc']:.4f} | precision: {metrics['precision']:.4f} | recall: {metrics['recall']:.4f} | mean_tp_iou: {metrics['mean_tp_iou']:.4f}",
				f"Pr@0.5: {metrics['pr_at_0.5']:.4f} | Pr@0.6: {metrics['pr_at_0.6']:.4f} | Pr@0.7: {metrics['pr_at_0.7']:.4f}",
				f"Pr@0.8: {metrics['pr_at_0.8']:.4f} | Pr@0.9: {metrics['pr_at_0.9']:.4f} | mIoU: {metrics['mIoU']:.4f} | cIoU: {metrics['cIoU']:.4f}",
				f"images with missing embedding: {metrics['missing_embed_images']}",
			],
		),
		encoding="utf-8",
	)

	if failed_lines:
		failed_txt.write_text("\n".join(failed_lines) + "\n", encoding="utf-8")
	else:
		failed_txt.write_text("\n", encoding="utf-8")

	if correct_lines:
		correct_txt.write_text("\n".join(correct_lines) + "\n", encoding="utf-8")
	else:
		correct_txt.write_text("\n", encoding="utf-8")

	pr_points = _compute_pr_curve(conf_tp_pairs, total_gt=int(stats["gt"]))
	with pr_points_csv.open("w", encoding="utf-8") as f:
		f.write("recall,precision\n")
		for recall_val, precision_val in pr_points:
			f.write(f"{recall_val:.6f},{precision_val:.6f}\n")

	_plot_metrics_overview(metrics, metrics_vis)
	_plot_pr_curve(pr_points, pr_curve_vis)

	return {
		"summary_json": summary_json,
		"summary_txt": summary_txt,
		"failed_txt": failed_txt,
		"correct_txt": correct_txt,
		"metrics_vis": metrics_vis,
		"pr_curve_vis": pr_curve_vis,
		"pr_points_csv": pr_points_csv,
	}


def main() -> None:
	args = parse_args()

	weights = Path(args.weights).expanduser().resolve()
	if not weights.exists():
		raise FileNotFoundError(f"Weights not found: {weights}")

	if int(args.text_embed_batch_size) <= 0:
		raise ValueError("--text-embed-batch-size must be > 0")

	data_yaml = _resolve_data_yaml(args)

	run_dir, vsi_dir = _resolve_run_dirs(args)

	data = check_det_dataset(str(data_yaml))
	split_source = data.get(args.split)
	if not split_source:
		raise ValueError(f"Split '{args.split}' is not available in {data_yaml}")
	prepared_root = _resolve_prepared_root(data, data_yaml)

	device = select_device(args.device, batch=args.batch)
	model = AutoBackend(weights=str(weights), device=device, dnn=False, data=str(data_yaml), fp16=args.half)
	if not (model.pt or model.nn_module):
		raise RuntimeError("This script only supports PyTorch .pt/.pth checkpoints for text-guided inference.")

	stride = int(model.stride.max()) if isinstance(model.stride, torch.Tensor) else int(model.stride)
	imgsz = check_imgsz(args.imgsz, stride=stride, max_dim=1)

	embedding_root = Path(args.embedding_dir).expanduser().resolve() if args.embedding_dir else prepared_root
	requested_split_emb = embedding_root / f"{args.split}_text_embeddings.pt"
	if args.no_auto_build_split_embedding:
		if not requested_split_emb.exists():
			raise FileNotFoundError(
				f"Missing split embedding: {requested_split_emb}. "
				"Remove --no-auto-build-split-embedding or generate it first."
			)
	else:
		text_device = str(args.device or "").strip() or "auto"
		_ensure_split_embedding(
			prepared_root=prepared_root,
			embedding_root=embedding_root,
			split=str(args.split),
			args=args,
			text_device=text_device,
		)

	expected_dim = int(args.text_embedding_dim) if int(args.text_embedding_dim) > 0 else _embedding_dim_from_model(model)
	embedding_store = ObjectTextEmbeddingStore(str(embedding_root), expected_dim=expected_dim)
	text_lookup = _load_text_lookup(embedding_root)

	dataset = TextGuidedYOLODataset(
		img_path=split_source,
		imgsz=imgsz,
		batch_size=args.batch,
		augment=False,
		hyp=DEFAULT_CFG,
		rect=True,
		cache=False,
		single_cls=True,
		stride=stride,
		pad=0.5,
		prefix=f"{args.split}: ",
		task="detect",
		classes=None,
		data=data,
		fraction=1.0,
	)
	dataloader = build_dataloader(dataset, batch=args.batch, workers=args.workers, shuffle=False, rank=-1)

	model.eval()
	model.warmup(imgsz=(1 if model.pt else args.batch, 3, imgsz, imgsz))

	stats = {
		"images": 0,
		"gt": 0,
		"pred": 0,
		"tp": 0,
		"fp": 0,
		"fn": 0,
		"missing_embed_images": 0,
		"tp_iou_sum": 0.0,
		"tp_iou_count": 0,
		"match_iou_sum": 0.0,
		"match_iou_count": 0,
		"match_ciou_sum": 0.0,
		"match_ciou_count": 0,
	}
	for thr in PR_IOU_THRESHOLDS:
		stats[f"pr_at_{thr:.1f}_count"] = 0
	failed_lines: List[str] = []
	correct_lines: List[str] = []
	conf_tp_pairs: List[Tuple[float, int]] = []

	max_images = int(args.max_images) if int(args.max_images) > 0 else int(1e18)

	progress = TQDM(dataloader, total=len(dataloader), desc=f"{args.split} visualize")
	for batch in progress:
		if stats["images"] >= max_images:
			break

		batch = _preprocess_batch(batch, model)
		sample_ids = _extract_sample_ids(batch)
		txt_vec, text_valid_mask, text_token_mask, text_phrase_weight, text_token_target_idx = embedding_store.get_batch_with_targets(
			sample_ids,
			device=batch["img"].device,
			dtype=batch["img"].dtype,
		)
		batch["txt_vec"] = txt_vec
		batch["text_valid_mask"] = text_valid_mask
		batch["text_token_mask"] = text_token_mask
		batch["text_phrase_weight"] = text_phrase_weight
		batch["text_token_target_idx"] = text_token_target_idx

		missing = int((~text_valid_mask).sum().item())
		if missing:
			if missing == len(sample_ids):
				preview = ", ".join(sample_ids[:8])
				raise RuntimeError(
					"All samples in the current batch are missing text embeddings. "
					f"Please verify sample IDs and embedding files. sample_ids[:8]={preview}"
				)
			LOGGER.warning(
				"Missing text embeddings for %d/%d samples in current batch.",
				missing,
				len(sample_ids),
			)

		with torch.inference_mode():
			raw_preds, phrase_logits_levels = _run_text_inference(model, batch)
			preds = non_max_suppression(
				raw_preds,
				conf_thres=float(args.conf),
				iou_thres=float(args.nms_iou),
				multi_label=False,
				agnostic=True,
				max_det=int(args.max_det),
			)

		for i, pred in enumerate(preds):
			if stats["images"] >= max_images:
				break

			im_file = Path(str(_batch_item(batch, "im_file", i) or "")).expanduser()
			if not im_file.exists():
				LOGGER.warning("Image not found for visualization: %s", im_file)
				continue

			image = cv2.imread(str(im_file))
			if image is None:
				LOGGER.warning("Failed to read image: %s", im_file)
				continue

			gt_xyxy = _xywhn_to_native_xyxy(batch, i)

			predn = pred.clone()
			if int(predn.shape[0]) > 0:
				scale_boxes(batch["img"].shape[2:], predn[:, :4], batch["ori_shape"][i], ratio_pad=batch["ratio_pad"][i])

			valid_tok = torch.nonzero(text_token_mask[i], as_tuple=False).view(-1)
			target_tok = text_token_target_idx[i, valid_tok] if int(valid_tok.numel()) > 0 else valid_tok.new_zeros((0,), dtype=torch.long)
			if int(valid_tok.numel()) > 0 and int(gt_xyxy.shape[0]) > 0:
				keep = (target_tok >= 0) & (target_tok < int(gt_xyxy.shape[0]))
				valid_tok = valid_tok[keep]
				target_tok = target_tok[keep]

			score = score_matrix_from_phrase_maps(
				phrase_logits_levels=phrase_logits_levels,
				image_index=i,
				token_indices=valid_tok,
				candidate_boxes_xyxy=pred[:, :4] if int(pred.shape[0]) > 0 else pred.new_zeros((0, 4)),
				input_hw=tuple(int(x) for x in batch["img"].shape[2:]),
			)
			row_ind, col_ind = assign_by_score_matrix(score, temperature=float(args.matching_temperature))
			assigned = {int(r): int(c) for r, c in zip(row_ind.tolist(), col_ind.tolist())}

			vis = image.copy()
			sample_id = sample_ids[i] if i < len(sample_ids) else im_file.stem
			desc_list = _resolve_description_list(batch, i, sample_id, text_lookup)
			if not desc_list:
				fallback_desc = _resolve_description(batch, i, sample_id, text_lookup)
				if fallback_desc:
					desc_list = [fallback_desc]
			gt_number_map: Dict[int, int] = {}
			for local_r, tgt in enumerate(target_tok.tolist()):
				if 0 <= int(tgt) < int(gt_xyxy.shape[0]):
					gt_number_map[int(tgt)] = int(local_r + 1)

			for gi, gt_box in enumerate(gt_xyxy.detach().cpu().tolist()):
				label_num = gt_number_map.get(gi, gi + 1)
				_draw_box(vis, gt_box, (255, 0, 0), [f"{label_num}"], label_anchor="br")

			n_pred = 0
			n_tp = 0
			n_fp = 0
			tp_iou_values: List[float] = []
			match_iou_values: List[float] = []
			match_ciou_values: List[float] = []
			total_desc = len(desc_list)
			object_is_tp: List[bool] = [False] * total_desc
			for local_r, _tok_idx in enumerate(valid_tok.tolist()):
				desc_idx = local_r
				if desc_idx >= total_desc:
					continue
				target_idx = int(target_tok[local_r].item()) if local_r < int(target_tok.numel()) else -1
				cand_idx = int(assigned.get(local_r, -1))
				if cand_idx < 0 or cand_idx >= int(predn.shape[0]):
					continue

				pred_box = predn[cand_idx, :4]
				conf = float(predn[cand_idx, 4].item())
				iou_val = 0.0
				ciou_val = 0.0
				if 0 <= target_idx < int(gt_xyxy.shape[0]):
					pred_box_1 = pred_box.view(1, 4)
					gt_box_1 = gt_xyxy[target_idx].view(1, 4)
					iou_val = float(box_iou_matrix(pred_box_1, gt_box_1)[0, 0].item())
					ciou_val = float(bbox_iou(pred_box_1, gt_box_1, xywh=False, CIoU=True).view(-1)[0].item())
					match_iou_values.append(iou_val)
					match_ciou_values.append(ciou_val)
					for thr in PR_IOU_THRESHOLDS:
						if iou_val >= float(thr):
							stats[f"pr_at_{thr:.1f}_count"] += 1

				is_tp = bool(iou_val >= float(args.match_iou))
				color = (0, 255, 0) if is_tp else (0, 0, 255)
				_draw_box(
					vis,
					pred_box.detach().cpu().tolist(),
					color,
					[
						f"{desc_idx + 1}",
						f"conf={conf:.2f}",
						f"IoU={iou_val:.2f}",
					],
					label_anchor="tl",
				)
				conf_tp_pairs.append((conf, 1 if is_tp else 0))
				n_pred += 1
				if is_tp:
					n_tp += 1
					object_is_tp[desc_idx] = True
					tp_iou_values.append(iou_val)
				else:
					n_fp += 1

			text_desc = _resolve_description(batch, i, sample_id, text_lookup)
			embedding_ok = bool(text_valid_mask[i].item()) if i < len(text_valid_mask) else False
			token_count = int(text_token_mask[i].sum().item()) if i < len(text_token_mask) else 0
			n_gt = int(gt_xyxy.shape[0])
			n_fn = max(total_desc - n_tp, 0)

			lines = [
				f"id={sample_id} | image={im_file.name}",
				f"embedding={'yes' if embedding_ok else 'no'} | tokens={token_count} | gt={n_gt} | pred={n_pred} | tp={n_tp} | fp={n_fp}",
				f"rules: GT=blue(id@br), Pred=green/red(id+conf+IoU@tl) | conf>={args.conf:.2f}, nms_iou={args.nms_iou:.2f}, match_iou={args.match_iou:.2f}",
			]
			if desc_list:
				lines.append("descriptions:")
				for di, dtext in enumerate(desc_list, start=1):
					wrapped = _wrap_description(dtext, image_width=vis.shape[1])
					if wrapped:
						first = wrapped[0].replace("text:", f"{di}.", 1)
						lines.append(first)
						for more in wrapped[1:]:
							lines.append(f"   {more.strip()}")
			else:
				lines.extend(_wrap_description(text_desc, image_width=vis.shape[1]))
			canvas = _header_canvas(vis, lines)

			safe_sid = _sanitize_name(sample_id)
			out_name = f"{stats['images'] + 1:06d}_{im_file.stem}_{safe_sid}.jpg"
			cv2.imwrite(str(vsi_dir / out_name), canvas)

			for desc_idx, obj_desc in enumerate(desc_list):
				desc_text = obj_desc if str(obj_desc).strip() else "<empty>"
				if object_is_tp[desc_idx]:
					correct_lines.append(desc_text)
				else:
					failed_lines.append(desc_text)

			stats["images"] += 1
			stats["gt"] += n_gt
			stats["pred"] += n_pred
			stats["tp"] += n_tp
			stats["fp"] += n_fp
			stats["fn"] += n_fn
			if not embedding_ok:
				stats["missing_embed_images"] += 1

			if n_tp > 0:
				stats["tp_iou_sum"] += float(sum(tp_iou_values))
				stats["tp_iou_count"] += len(tp_iou_values)

			if match_iou_values:
				stats["match_iou_sum"] += float(sum(match_iou_values))
				stats["match_iou_count"] += len(match_iou_values)

			if match_ciou_values:
				stats["match_ciou_sum"] += float(sum(match_ciou_values))
				stats["match_ciou_count"] += len(match_ciou_values)

		progress.set_postfix(images=stats["images"], tp=stats["tp"], fp=stats["fp"])

	acc = stats["tp"] / max(stats["tp"] + stats["fp"] + stats["fn"], 1)
	precision = stats["tp"] / max(stats["pred"], 1)
	recall = stats["tp"] / max(stats["gt"], 1)
	mean_tp_iou = stats["tp_iou_sum"] / max(stats["tp_iou_count"], 1)
	m_iou = stats["match_iou_sum"] / max(stats["match_iou_count"], 1)
	c_iou = stats["match_ciou_sum"] / max(stats["match_ciou_count"], 1)
	pr_at_values = {thr: stats[f"pr_at_{thr:.1f}_count"] / max(stats["pred"], 1) for thr in PR_IOU_THRESHOLDS}
	artifact_paths = _write_artifacts(
		run_dir=run_dir,
		vsi_dir=vsi_dir,
		stats=stats,
		weights=weights,
		data_yaml=data_yaml,
		embedding_root=embedding_root,
		args=args,
		failed_lines=failed_lines,
		correct_lines=correct_lines,
		conf_tp_pairs=conf_tp_pairs,
	)

	print("\n=== Text-Guided Validation Summary ===")
	print(f"weights: {weights}")
	print(f"data: {data_yaml}")
	print(f"split: {args.split}")
	print(f"embedding root: {embedding_root}")
	print(f"run dir: {run_dir}")
	print(f"saved visualizations: {vsi_dir}")
	print(f"images: {stats['images']}")
	print(f"total gt: {stats['gt']}")
	print(f"total pred: {stats['pred']}")
	print(f"tp: {stats['tp']} | fp: {stats['fp']} | fn: {stats['fn']}")
	print(f"acc: {acc:.4f} | precision: {precision:.4f} | recall: {recall:.4f} | mean_tp_iou: {mean_tp_iou:.4f}")
	print(
		" | ".join([f"Pr@{thr:.1f}: {pr_at_values[thr]:.4f}" for thr in PR_IOU_THRESHOLDS])
	)
	print(f"mIoU: {m_iou:.4f} | cIoU: {c_iou:.4f}")
	print(f"images with missing embedding: {stats['missing_embed_images']}")
	print(f"metrics summary txt: {artifact_paths['summary_txt']}")
	print(f"metrics summary json: {artifact_paths['summary_json']}")
	print(f"failed object txt: {artifact_paths['failed_txt']}")
	print(f"correct object txt: {artifact_paths['correct_txt']}")
	print(f"metrics visualization: {artifact_paths['metrics_vis']}")
	print(f"pr curve visualization: {artifact_paths['pr_curve_vis']}")


if __name__ == "__main__":
	main()
