import argparse
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from dataset.utils import prepare_dataset
from dataset.voc_object_dataset import load_sample_rows
from lib.general import print_metrics, resolve_device, resolve_local_weights
from text_encoder import TextGuidedDetectionValidator, configure_text_guidance
from text_encoder.embedding_store import ObjectTextEmbeddingStore


def _parse_phrase_types(raw: str) -> list[str]:
    values = [x.strip().upper() for x in str(raw or "").split(",") if x.strip()]
    return values or ["NP", "PP", "ADJP"]


def _parse_phrase_weight_string(raw: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in str(raw or "").split(","):
        part = item.strip()
        if not part or ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().upper()
        value = value.strip()
        if not key:
            continue
        try:
            score = float(value)
        except ValueError:
            continue
        if score > 0:
            weights[key] = score
    return weights


def _supports_text_conditioning(model) -> bool:
    """Return True when model exposes txt_vec in predict/forward signature."""
    for fn_name in ("predict", "forward"):
        fn = getattr(model, fn_name, None)
        if not callable(fn):
            continue
        try:
            if "txt_vec" in inspect.signature(fn).parameters:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _resolve_eval_postprocess(args: argparse.Namespace) -> Tuple[str, float, int]:
    """Resolve NMS-related eval knobs so offline val can match training-time metrics."""
    eval_mode = str(args.eval_mode).strip().lower()
    if eval_mode == "train-compatible":
        default_conf, default_max_det = 0.001, 300
    elif eval_mode == "grounding-top1":
        default_conf, default_max_det = 0.25, 1
    else:
        raise ValueError(f"Unsupported --eval-mode: {args.eval_mode}")

    conf = float(default_conf if args.conf is None else args.conf)
    max_det_raw = default_max_det if args.max_det is None else args.max_det
    max_det = int(max(1, int(max_det_raw)))
    return eval_mode, conf, max_det


def _load_checkpoint_context(weights_path: str) -> Dict[str, Any]:
    """Best-effort load of checkpoint metadata for comparability diagnostics."""
    try:
        ckpt = torch.load(weights_path, map_location="cpu")
    except Exception as e:
        LOGGER.warning(f"Unable to read checkpoint metadata from {weights_path}: {e}")
        return {}

    if not isinstance(ckpt, dict):
        return {}

    train_args = ckpt.get("train_args", {})
    train_metrics = ckpt.get("train_metrics", {})
    return {
        "epoch": ckpt.get("epoch", None),
        "best_fitness": ckpt.get("best_fitness", None),
        "train_args": train_args if isinstance(train_args, dict) else {},
        "train_metrics": train_metrics if isinstance(train_metrics, dict) else {},
    }


def _print_checkpoint_diagnostics(ckpt_ctx: Dict[str, Any], data_path: str, conf_thres: float, max_det: int) -> None:
    """Print checkpoint-side metrics/args so users can compare train-time and offline val settings."""
    if not ckpt_ctx:
        return

    epoch = ckpt_ctx.get("epoch", None)
    best_fitness = ckpt_ctx.get("best_fitness", None)
    if epoch is not None:
        LOGGER.info(f"Checkpoint metadata: epoch={epoch}, best_fitness={best_fitness}")

    train_metrics = ckpt_ctx.get("train_metrics", {})
    if isinstance(train_metrics, dict) and train_metrics:
        metric_keys = (
            "fitness",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        )
        parts = []
        for key in metric_keys:
            if key not in train_metrics:
                continue
            value = train_metrics.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{key}={float(value):.6f}")
            else:
                parts.append(f"{key}={value}")
        if parts:
            LOGGER.info("Checkpoint train_metrics snapshot: " + ", ".join(parts))

    train_args = ckpt_ctx.get("train_args", {})
    if isinstance(train_args, dict) and train_args:
        train_conf = train_args.get("conf", None)
        if train_conf is None:
            train_conf = 0.001

        train_data = str(train_args.get("data", "") or "").strip()
        train_iou = train_args.get("iou", None)
        train_max_det = train_args.get("max_det", None)
        train_single_cls = train_args.get("single_cls", None)
        train_agnostic_nms = train_args.get("agnostic_nms", None)

        LOGGER.info(
            "Train args snapshot: "
            f"data={train_data or '<empty>'}, conf={train_conf}, iou={train_iou}, "
            f"max_det={train_max_det}, single_cls={train_single_cls}, agnostic_nms={train_agnostic_nms}"
        )
        LOGGER.info(
            "Current offline val settings: "
            f"data={data_path}, conf={conf_thres}, max_det={max_det}, single_cls=True, agnostic_nms=True"
        )

        if train_data and train_data != data_path:
            LOGGER.warning(
                "Current --data path differs from checkpoint train_args['data']. "
                "This often causes large metric deltas across train/offline val."
            )


def _resolve_split_rows_file(data_yaml_path: str, split: str = "val") -> Path | None:
    """Resolve split rows file path (jsonl/txt) from data.yaml with root path handling."""
    yaml_path = Path(data_yaml_path).expanduser().resolve()
    if not yaml_path.exists():
        LOGGER.warning(f"Data YAML not found for embedding coverage check: {yaml_path}")
        return None

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f) or {}
    except Exception as e:
        LOGGER.warning(f"Failed to read data yaml for embedding coverage check: {e}")
        return None

    raw_split = data_cfg.get(split)
    if not isinstance(raw_split, str) or not raw_split.strip():
        LOGGER.warning(f"data.yaml has no usable '{split}' entry for embedding coverage check.")
        return None

    raw_root = str(data_cfg.get("path", "") or "").strip()
    if raw_root:
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            root = (yaml_path.parent / root).resolve()
        else:
            root = root.resolve()
    else:
        root = yaml_path.parent

    split_path = Path(raw_split).expanduser()
    if not split_path.is_absolute():
        split_path = (root / split_path).resolve()
    else:
        split_path = split_path.resolve()

    if not split_path.exists():
        LOGGER.warning(f"Split rows file not found for embedding coverage check: {split_path}")
        return None

    return split_path


def _collect_sample_ids(rows: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id", "") or row.get("id", "") or "").strip()
        if not sample_id:
            label_ref = str(row.get("label", "") or row.get("label_file", "") or "").strip()
            if label_ref:
                sample_id = Path(label_ref).stem
        if not sample_id:
            image_ref = str(row.get("image", "") or row.get("im_file", "") or "").strip()
            if image_ref:
                sample_id = Path(image_ref).stem
        if sample_id:
            ids.append(sample_id)
    return ids


def _report_embedding_coverage(data_path: str, embedding_root: Path, expected_dim: int) -> None:
    """Report val-split embedding coverage before running validation."""
    split_rows = _resolve_split_rows_file(data_path, split="val")
    if split_rows is None:
        return
    if split_rows.suffix.lower() != ".jsonl":
        LOGGER.warning(
            f"Embedding coverage check currently supports jsonl rows files only, got: {split_rows.name}. "
            "Skipping pre-check."
        )
        return

    try:
        rows = load_sample_rows(str(split_rows))
    except Exception as e:
        LOGGER.warning(f"Failed to load val samples for embedding coverage check: {e}")
        return

    sample_ids = _collect_sample_ids(rows)
    if not sample_ids:
        LOGGER.warning("No sample_ids found in val rows file; skipping embedding coverage check.")
        return

    try:
        store = ObjectTextEmbeddingStore(embedding_root=str(embedding_root), expected_dim=int(expected_dim))
    except Exception as e:
        LOGGER.warning(f"Failed to build embedding store for coverage check: {e}")
        return

    valid_total = 0
    missing_preview: List[str] = []
    chunk = 256
    cpu = torch.device("cpu")

    for start in range(0, len(sample_ids), chunk):
        sub_ids = sample_ids[start : start + chunk]
        _, valid_mask, _, _ = store.get_batch(sub_ids, device=cpu, dtype=torch.float32)
        valid_list = valid_mask.detach().cpu().tolist()
        valid_total += int(sum(bool(x) for x in valid_list))
        if len(missing_preview) < 8:
            for sid, ok in zip(sub_ids, valid_list):
                if not bool(ok):
                    missing_preview.append(sid)
                if len(missing_preview) >= 8:
                    break

    total = len(sample_ids)
    coverage = (valid_total / total) if total else 0.0
    LOGGER.info(
        f"Embedding coverage check (val split): {valid_total}/{total} ({coverage:.2%}), "
        f"embedding files loaded={len(store.loaded_files)}, dim={store.dim}"
    )
    if valid_total < total:
        preview = ", ".join(missing_preview) if missing_preview else "<none>"
        LOGGER.warning(
            f"Missing embeddings for {total - valid_total} validation sample(s). "
            f"Example missing sample_id(s): {preview}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a text-guided YOLOv12 model with current project settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", "--model", dest="weights", type=str, default="./runs/train/yolov12-custom/weights/best.pt", help="Model or checkpoint path.")
    parser.add_argument("--data", type=str, default="", help="Optional existing data.yaml path. If set, skip dataset preparation.")
    parser.add_argument("--voc-root", type=str, default="/data/bxc/DIOR-RSVG", help="VOC root with JPGEImages/JPEGImages, Annotations and split txt files.")
    parser.add_argument("--images-dir", type=str, default="JPGEImages", help="Image folder name under VOC root.")
    parser.add_argument("--annotations-dir", type=str, default="Annotations", help="XML annotation folder name.")
    parser.add_argument("--train-list", type=str, default="train.txt", help="Train split txt file name.")
    parser.add_argument("--val-list", type=str, default="val.txt", help="Val split txt file name.")
    parser.add_argument("--test-list", type=str, default="test.txt", help="Test split txt file name.")
    parser.add_argument("--prepared-dir", "--prepared_dir", dest="prepared_dir", type=str, default="pre_datasets/DIOR_RSVG", help="Output dir for generated data.yaml and mapping files.")

    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", type=str, default="0", help="Device, e.g. '0', '0,1', 'cpu' or 'auto'.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader worker count.")
    parser.add_argument("--project", type=str, default="runs/val", help="Project directory.")
    parser.add_argument("--name", type=str, default="yolov12-custom-val", help="Experiment name.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse existing experiment directory.")
    parser.add_argument("--save-json", action="store_true", help="Save COCO JSON during validation.")
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=("train-compatible", "grounding-top1"),
        default="train-compatible",
        help="Postprocess preset. train-compatible aligns with training-time mAP validation; grounding-top1 is strict top-1 grounding.",
    )
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold for NMS. Empty means use --eval-mode default.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--max-det", dest="max_det", type=int, default=None, help="Maximum detections per image. Empty means use --eval-mode default.")
    parser.add_argument(
        "--multi-class-eval",
        action="store_true",
        help="Debug only: disable class-agnostic grounding postprocess and keep multi-class behavior.",
    )

    parser.add_argument("--text-embedding-dir", type=str, default="", help="Directory that contains *_text_embeddings.pt files. Empty means --prepared-dir.")
    parser.add_argument("--text-embedding-dim", type=int, default=768, help="Input text embedding dimension.")
    parser.add_argument("--text-phrase-types", type=str, default="NP,PP,ADJP", help="Phrase types used in preprocessing, comma-separated.")
    parser.add_argument(
        "--text-phrase-type-weights",
        type=str,
        default="NP:1.0,PP:1.2,ADJP:0.8,FALLBACK:1.0",
        help="Phrase type priors for weighted phrase aggregation.",
    )
    parser.add_argument(
        "--text-aggregation-mode",
        type=str,
        choices=("weighted_sum", "mean"),
        default="weighted_sum",
        help="Phrase-to-heatmap aggregation strategy.",
    )
    parser.add_argument("--text-guidance-strength", type=float, default=0.25, help="Feature modulation strength for text guidance.")
    parser.add_argument("--text-cls-gate-strength", type=float, default=0.8, help="Logit gating strength applied to the classification branch.")
    parser.add_argument("--text-alignment-temperature", type=float, default=0.07, help="Temperature used in phrase-visual similarity computation.")
    parser.add_argument("--text-lambda-heatmap", type=float, default=0.3, help="Phrase contrastive supervision weight.")
    parser.add_argument("--text-seq-enhance", action="store_true", help="Enable lightweight text sequence enhancement before similarity.")
    parser.add_argument("--text-seq-conv-layers", type=int, default=1, help="Number of 1D conv residual layers for text sequence enhancement.")
    parser.add_argument("--text-seq-kernel-size", type=int, default=3, help="Kernel size for text sequence 1D conv.")
    parser.add_argument("--text-seq-dropout", type=float, default=0.0, help="Dropout ratio inside text sequence enhancement block.")
    parser.add_argument(
        "--text-seq-pooling-mode",
        type=str,
        choices=("none", "learnable_weight"),
        default="none",
        help="Adaptive token weighting mode used before phrase aggregation.",
    )
    parser.add_argument("--text-seq-pool-temperature", type=float, default=1.0, help="Temperature for adaptive token weighting.")
    parser.add_argument("--visual-attr-enabled", action="store_true", help="Enable normalized geometry/stat attributes on visual features before similarity.")
    parser.add_argument("--visual-attr-scale", type=float, default=1.0, help="Scale factor for visual attribute channels.")
    parser.add_argument("--visual-attr-eps", type=float, default=1e-6, help="Numerical epsilon used in visual attribute normalization.")
    parser.add_argument("--text-multi-proj-enabled", action="store_true", help="Enable three parallel text-visual projections (geo/attr/sem) before similarity.")
    parser.add_argument("--text-multi-proj-score-scale", type=float, default=1.0, help="Scale factor applied after summing three projection scores.")
    parser.add_argument("--text-orth-loss-weight", type=float, default=0.05, help="Weight for orthogonality regularizer across three visual projections.")
    parser.add_argument("--text-keep-augmentations", action="store_false", dest="text_disable_augmentations", help="Keep original augmentations in text-guided pipeline.")
    parser.add_argument("--visualize", action="store_true", help="Save per-image visualization overlays during validation.")
    parser.add_argument(
        "--skip-embedding-coverage-check",
        action="store_true",
        help="Skip pre-validation embedding coverage diagnostics on val split.",
    )
    return parser.parse_args()


class VisualizingTextGuidedDetectionValidator(TextGuidedDetectionValidator):
    """Text-guided validator that saves per-image visualization and precision@0.50."""

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, embedding_store=None):
        super().__init__(
            dataloader=dataloader,
            save_dir=save_dir,
            pbar=pbar,
            args=args,
            _callbacks=_callbacks,
            embedding_store=embedding_store,
        )
        self.vis_dir = self.save_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.total_pred_050 = 0
        self.correct_pred_050 = 0

    @staticmethod
    def _clamp_box(box, width: int, height: int):
        x1 = int(max(0, min(width - 1, round(float(box[0])))))
        y1 = int(max(0, min(height - 1, round(float(box[1])))))
        x2 = int(max(0, min(width - 1, round(float(box[2])))))
        y2 = int(max(0, min(height - 1, round(float(box[3])))))
        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)
        return x1, y1, x2, y2

    @staticmethod
    def _draw_header(image, lines):
        if not lines:
            return image
        line_height = 24
        pad = 8
        header_h = min(image.shape[0], pad * 2 + line_height * len(lines))
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1] - 1, header_h), (24, 24, 24), thickness=-1)
        image = cv2.addWeighted(overlay, 0.68, image, 0.32, 0.0)
        for idx, line in enumerate(lines):
            y = pad + (idx + 1) * line_height - 8
            if y >= header_h:
                break
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA)
        return image

    def _save_visualization(self, image_path: str, gt_boxes: torch.Tensor, predn: torch.Tensor, correct_050: torch.Tensor):
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.warning(f"Could not read image for visualization: {image_path}")
            return

        height, width = image.shape[:2]
        gt_cpu = gt_boxes.detach().cpu() if isinstance(gt_boxes, torch.Tensor) else torch.zeros((0, 4))
        pred_cpu = predn.detach().cpu() if isinstance(predn, torch.Tensor) else torch.zeros((0, 6))
        correct_cpu = correct_050.detach().cpu() if isinstance(correct_050, torch.Tensor) else torch.zeros(0, dtype=torch.bool)

        for gt_box in gt_cpu:
            x1, y1, x2, y2 = self._clamp_box(gt_box[:4], width, height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, "GT", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        pred_count = int(pred_cpu.shape[0])
        correct_count = int(correct_cpu.sum().item()) if correct_cpu.numel() else 0
        wrong_count = pred_count - correct_count
        image_precision = (correct_count / pred_count) if pred_count else 0.0
        overall_precision = (self.correct_pred_050 / self.total_pred_050) if self.total_pred_050 else 0.0

        for idx, pred_box in enumerate(pred_cpu):
            is_correct = bool(correct_cpu[idx].item()) if idx < correct_cpu.numel() else False
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            x1, y1, x2, y2 = self._clamp_box(pred_box[:4], width, height)
            conf = float(pred_box[4])
            status = "OK" if is_correct else "ERR"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"P {status} {conf:.2f}",
                (x1, min(height - 5, max(12, y1 - 5))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        lines = [
            f"Image: {Path(image_path).name}",
            f"GT(blue): {int(gt_cpu.shape[0])} | Pred: {pred_count} | Correct(green): {correct_count} | Wrong(red): {wrong_count}",
            f"Precision@0.50 (image): {image_precision:.4f} | Precision@0.50 (overall): {overall_precision:.4f}",
        ]
        image = self._draw_header(image, lines)

        save_name = f"{self.batch_i:05d}_{Path(image_path).stem}_vis.jpg"
        save_path = self.vis_dir / save_name
        cv2.imwrite(str(save_path), image)

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()

            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)

                empty_pred = torch.zeros((0, 6), dtype=torch.float32, device=self.device)
                empty_correct = torch.zeros(0, dtype=torch.bool, device=self.device)
                self._save_visualization(batch["im_file"][si], bbox, empty_pred, empty_correct)
                continue

            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

            correct_050 = stat["tp"][:, 0]
            self.total_pred_050 += int(npr)
            self.correct_pred_050 += int(correct_050.sum().item())
            self._save_visualization(batch["im_file"][si], bbox, predn, correct_050)

    def get_stats(self):
        stats = super().get_stats()
        precision_050 = (self.correct_pred_050 / self.total_pred_050) if self.total_pred_050 else 0.0
        stats["metrics/precision_visual@0.50"] = float(precision_050)
        return stats

    def finalize_metrics(self, *args, **kwargs):
        super().finalize_metrics(*args, **kwargs)
        precision_050 = (self.correct_pred_050 / self.total_pred_050) if self.total_pred_050 else 0.0
        LOGGER.info(
            f"Saved visualization images to {self.vis_dir}. "
            f"Overall Precision@0.50: {precision_050:.4f} ({self.correct_pred_050}/{self.total_pred_050})"
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    eval_mode, conf_thres, max_det = _resolve_eval_postprocess(args)

    data_path = str(Path(args.data).expanduser().resolve()) if str(args.data).strip() else prepare_dataset(args, device)
    weights_path = resolve_local_weights(args.weights)
    print(f"Using weights: {weights_path}")
    print(f"Using device: {device}")

    ckpt_ctx = _load_checkpoint_context(weights_path)
    _print_checkpoint_diagnostics(ckpt_ctx, data_path, conf_thres, max_det)

    model = YOLO(weights_path)
    if not _supports_text_conditioning(model.model):
        raise RuntimeError(
            "Loaded checkpoint does not support txt_vec/txt_token_mask inference. "
            f"Please use a text-guided trained checkpoint, got: {weights_path}"
        )

    embedding_root = Path(args.text_embedding_dir or args.prepared_dir).expanduser().resolve()
    phrase_types = _parse_phrase_types(args.text_phrase_types)
    phrase_type_weights = _parse_phrase_weight_string(args.text_phrase_type_weights)
    configure_text_guidance(
        {
            "enabled": True,
            "embedding_dir": str(embedding_root),
            "embedding_dim": int(args.text_embedding_dim),
            "phrase_types": phrase_types,
            "phrase_type_weights": phrase_type_weights,
            "aggregation_mode": str(args.text_aggregation_mode),
            "guidance_strength": float(args.text_guidance_strength),
            "cls_gate_strength": float(args.text_cls_gate_strength),
            "alignment_temperature": float(args.text_alignment_temperature),
            "lambda_heatmap": float(args.text_lambda_heatmap),
            "contrastive_loss_type": "logsigmoid_margin",
            "disable_augmentations": bool(args.text_disable_augmentations),
            "text_seq_enhance": bool(args.text_seq_enhance),
            "text_seq_conv_layers": int(args.text_seq_conv_layers),
            "text_seq_kernel_size": int(args.text_seq_kernel_size),
            "text_seq_dropout": float(args.text_seq_dropout),
            "text_seq_pooling_mode": str(args.text_seq_pooling_mode),
            "text_seq_pool_temperature": float(args.text_seq_pool_temperature),
            "visual_attr_enabled": bool(args.visual_attr_enabled),
            "visual_attr_scale": float(args.visual_attr_scale),
            "visual_attr_eps": float(args.visual_attr_eps),
            "multi_proj_enabled": bool(args.text_multi_proj_enabled),
            "multi_proj_score_scale": float(args.text_multi_proj_score_scale),
            "orth_loss_weight": float(args.text_orth_loss_weight),
        }
    )
    print(f"Text guidance enabled, embedding root: {embedding_root}")
    print(
        "Text phrase config: "
        f"types={phrase_types}, aggregation={args.text_aggregation_mode}, "
        f"type_weights={phrase_type_weights if phrase_type_weights else '<default>'}"
    )
    if not args.skip_embedding_coverage_check:
        _report_embedding_coverage(
            data_path=data_path,
            embedding_root=embedding_root,
            expected_dim=int(args.text_embedding_dim),
        )

    val_kwargs: Dict[str, Any] = {
        "data": data_path,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "save_json": args.save_json,
        "device": device,
        "conf": conf_thres,
        "iou": float(args.iou),
        "max_det": max_det,
        "single_cls": not bool(args.multi_class_eval),
        "agnostic_nms": not bool(args.multi_class_eval),
    }

    if args.multi_class_eval:
        print(
            f"Validation postprocess mode: multi-class debug mode "
            f"(eval_mode={eval_mode}, conf={conf_thres:.4f}, max_det={max_det})."
        )
    else:
        print(
            f"Validation postprocess mode: grounding class-agnostic "
            f"(eval_mode={eval_mode}, single_cls=True, agnostic_nms=True, conf={conf_thres:.4f}, max_det={max_det})."
        )

    comparable_to_train = (
        (not bool(args.multi_class_eval))
        and eval_mode == "train-compatible"
        and conf_thres <= 0.01
        and max_det >= 300
    )
    if comparable_to_train:
        print("Validation metric mode: train-compatible (expected to match training-time val settings).")
    else:
        LOGGER.warning(
            "Current validation settings are not directly comparable to training-time val metrics "
            "(typically conf=0.001, single_cls=True, agnostic_nms=True, max_det=300)."
        )

    validator_cls = TextGuidedDetectionValidator
    if args.visualize:
        LOGGER.warning("Current phase-2 set matching path uses TextGuidedDetectionValidator; --visualize overlays are temporarily disabled.")
    print(f"Validator: {validator_cls.__name__}")
    print("Starting validation with text guidance...")
    val_metrics = model.val(validator=validator_cls, **val_kwargs)
    print_metrics("Validation Metrics", val_metrics)
    if isinstance(val_metrics, dict) and args.visualize:
        precision_050 = float(val_metrics.get("metrics/precision_visual@0.50", 0.0))
        print(f"Overall Precision@0.50 (visual): {precision_050:.4f}")


if __name__ == "__main__":
    main()
