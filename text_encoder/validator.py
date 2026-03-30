from __future__ import annotations

from pathlib import Path
from typing import List

import torch

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER, ops

from .embedding_store import ObjectTextEmbeddingStore
from .matching import assign_by_score_matrix, box_iou_matrix, score_matrix_from_phrase_maps
from .settings import get_text_guidance_config


class TextGuidedDetectionValidator(DetectionValidator):
    """Detection validator that injects per-image text embeddings before inference."""

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        pbar=None,
        args=None,
        _callbacks=None,
        embedding_store: ObjectTextEmbeddingStore | None = None,
    ):
        super().__init__(dataloader=dataloader, save_dir=save_dir, pbar=pbar, args=args, _callbacks=_callbacks)
        self.text_cfg = get_text_guidance_config()
        self.text_enabled = bool(self.text_cfg.get("enabled", False))
        self.embedding_store = embedding_store
        self._embedding_store_reported = embedding_store is not None

        self.strict_grounding = bool(self.text_cfg.get("strict_grounding", True))
        self.strict_match_iou = float(self.text_cfg.get("strict_match_iou", 0.5))
        self.matching_temperature = float(max(1e-3, self.text_cfg.get("matching_temperature", 0.7)))
        self.strict_nms_conf = float(max(0.0, self.text_cfg.get("strict_nms_conf", 0.25)))
        self.strict_max_candidates_factor = int(max(1, self.text_cfg.get("strict_max_candidates_factor", 4)))

        # Text-guided grounding is class-agnostic by design.
        self.args.single_cls = True
        self.args.agnostic_nms = True

        self.strict_total_desc = 0
        self.strict_pred_desc = 0
        self.strict_correct = 0
        self.strict_misattribution = 0
        self.strict_hallucination = 0
        self.strict_miss = 0
        self._strict_phrase_logits: List[torch.Tensor] = []

    def __call__(self, trainer=None, model=None):
        if trainer is not None:
            self.text_enabled = bool(getattr(trainer, "text_enabled", self.text_enabled))
            trainer_store = getattr(trainer, "embedding_store", None)
            if trainer_store is not None:
                self.embedding_store = trainer_store
                self._embedding_store_reported = True
        return super().__call__(trainer=trainer, model=model)

    def _resolve_embedding_root(self) -> str:
        configured = str(self.text_cfg.get("embedding_dir", "") or "").strip()
        if configured:
            return configured

        data_root = str((self.data or {}).get("path", "") or "").strip()
        if data_root:
            return str(Path(data_root).expanduser().resolve())

        raise ValueError("Unable to resolve embedding root for validation. Set --text-embedding-dir explicitly.")

    def _ensure_embedding_store(self) -> None:
        if not self.text_enabled or self.embedding_store is not None:
            return

        embedding_root = self._resolve_embedding_root()
        self.embedding_store = ObjectTextEmbeddingStore(
            embedding_root=embedding_root,
            expected_dim=int(self.text_cfg.get("embedding_dim", 768)),
        )

        if not self._embedding_store_reported:
            LOGGER.info(
                f"Text-guided validation enabled with {len(self.embedding_store.loaded_files)} embedding file(s), "
                f"dim={self.embedding_store.dim}"
            )
            self._embedding_store_reported = True

    def _extract_sample_ids(self, batch) -> List[str]:
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

    @staticmethod
    def _prepare_backend_img(backend, img: torch.Tensor) -> torch.Tensor:
        """Apply backend-required dtype/layout transforms before calling wrapped PyTorch model."""
        if getattr(backend, "fp16", False) and img.dtype != torch.float16:
            img = img.half()
        if getattr(backend, "nhwc", False):
            img = img.permute(0, 2, 3, 1)
        return img

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if not self.text_enabled:
            return batch

        self._ensure_embedding_store()
        if self.embedding_store is None:
            return batch

        sample_ids = self._extract_sample_ids(batch)
        txt_vec, text_valid_mask, text_token_mask, text_phrase_weight, text_token_target_idx = self.embedding_store.get_batch_with_targets(
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
            valid_list = text_valid_mask.detach().cpu().tolist()
            missing_ids = [sample_ids[i] for i, ok in enumerate(valid_list) if not ok]
            preview = ", ".join(missing_ids[:5]) if missing_ids else "<none>"
            message = (
                f"Missing text embeddings for {missing}/{len(sample_ids)} validation sample(s). "
                f"Sample IDs (first {min(5, len(missing_ids))}): {preview}"
            )
            if missing == len(sample_ids):
                raise RuntimeError(
                    message
                    + ". Validation would degrade to image-only YOLO. "
                    "Please verify sample_id/label_file naming matches embedding IDs."
                )
            LOGGER.warning(message)

        # Force bounding box classes to 0 for class-agnostic YOLO detection head
        if "cls" in batch and batch["cls"] is not None:
            batch["cls"] = torch.zeros_like(batch["cls"])

        return batch

    def postprocess(self, preds):
        """Apply grounding-friendly NMS: class-agnostic and single-label."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=False,
            agnostic=True,
            max_det=max(1, int(self.args.max_det)),
        )

    def inference(self, model, batch, augment=False):
        txt_vec = batch.get("txt_vec")
        txt_token_mask = batch.get("text_token_mask")
        txt_phrase_weight = batch.get("text_phrase_weight")
        if not isinstance(txt_vec, torch.Tensor):
            raise RuntimeError("Text-guided validation requires batch['txt_vec'], but it is missing.")

        infer_model = model
        img = batch["img"]
        if hasattr(model, "model") and hasattr(model, "pt"):
            if not (getattr(model, "pt", False) or getattr(model, "nn_module", False)):
                raise RuntimeError(
                    "Text-guided validation requires a PyTorch backend (.pt/.pth). "
                    "Current backend does not support text-conditioned inputs."
                )
            img = self._prepare_backend_img(model, img)
            infer_model = model.model

        try:
            out = infer_model(
                img,
                augment=augment,
                txt_vec=txt_vec,
                txt_token_mask=txt_token_mask,
                txt_phrase_weight=txt_phrase_weight,
            )
            self._strict_phrase_logits = []
            if self.strict_grounding and hasattr(infer_model, "_last_text_outputs"):
                last = getattr(infer_model, "_last_text_outputs", None)
                if isinstance(last, dict) and isinstance(last.get("phrase_logits"), list):
                    self._strict_phrase_logits = last.get("phrase_logits")
            return out
        except TypeError as e:
            retry_on_wrapped_model = (
                infer_model is model
                and ("txt_token_mask" in str(e) or "txt_phrase_weight" in str(e))
                and hasattr(model, "model")
                and callable(getattr(model, "model"))
                and (getattr(model, "pt", False) or getattr(model, "nn_module", False))
            )
            if retry_on_wrapped_model:
                retry_img = self._prepare_backend_img(model, batch["img"])
                try:
                    out = model.model(
                        retry_img,
                        augment=augment,
                        txt_vec=txt_vec,
                        txt_token_mask=txt_token_mask,
                        txt_phrase_weight=txt_phrase_weight,
                    )
                    self._strict_phrase_logits = []
                    if self.strict_grounding and hasattr(model.model, "_last_text_outputs"):
                        last = getattr(model.model, "_last_text_outputs", None)
                        if isinstance(last, dict) and isinstance(last.get("phrase_logits"), list):
                            self._strict_phrase_logits = last.get("phrase_logits")
                    return out
                except TypeError as retry_e:
                    raise TypeError(
                        "Current model does not accept txt_vec/txt_token_mask. "
                        "Please use the text-guided YOLOv12 model definition and checkpoint."
                    ) from retry_e

            raise TypeError(
                "Current model does not accept txt_vec/txt_token_mask. "
                "Please use the text-guided YOLOv12 model definition and checkpoint."
            ) from e

    def init_metrics(self, model):
        super().init_metrics(model)
        self.strict_total_desc = 0
        self.strict_pred_desc = 0
        self.strict_correct = 0
        self.strict_misattribution = 0
        self.strict_hallucination = 0
        self.strict_miss = 0

    def update_metrics(self, preds, batch):
        if not self.strict_grounding:
            return super().update_metrics(preds, batch)

        txt_vec = batch.get("txt_vec")
        txt_mask = batch.get("text_token_mask")
        txt_target = batch.get("text_token_target_idx")
        if not isinstance(txt_vec, torch.Tensor) or not isinstance(txt_target, torch.Tensor):
            raise RuntimeError("Strict grounding requires txt_vec and text_token_target_idx in batch.")

        if isinstance(preds, list):
            # `postprocess()` already converted raw outputs to NMS-format [N, 6].
            nms_preds = preds
        else:
            nms_preds = ops.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                labels=self.lb,
                multi_label=False,
                agnostic=True,
                max_det=max(1, int(self.args.max_det)),
            )

        bsz = int(txt_vec.shape[0])
        if isinstance(txt_mask, torch.Tensor) and txt_mask.ndim == 2 and int(txt_mask.shape[0]) == bsz:
            token_mask = txt_mask.to(device=txt_vec.device, dtype=torch.bool)
        else:
            token_mask = torch.ones((bsz, int(txt_vec.shape[1])), device=txt_vec.device, dtype=torch.bool)

        input_hw = tuple(int(x) for x in batch["img"].shape[2:])
        for si in range(bsz):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            gt_bbox = pbatch.get("bbox")
            if not isinstance(gt_bbox, torch.Tensor):
                gt_bbox = torch.zeros((0, 4), device=self.device)
            gt_n = int(gt_bbox.shape[0]) if isinstance(gt_bbox, torch.Tensor) else 0

            valid_tok = torch.nonzero(token_mask[si], as_tuple=False).view(-1)
            if int(valid_tok.numel()) > 0:
                token_targets = txt_target[si, valid_tok]
                if gt_n > 0:
                    keep_tok = (token_targets >= 0) & (token_targets < gt_n)
                else:
                    keep_tok = token_targets >= 0
                if bool(keep_tok.any()):
                    valid_tok = valid_tok[keep_tok]
                else:
                    valid_tok = valid_tok[:0]

            k = int(valid_tok.numel())
            if k <= 0:
                continue

            self.strict_total_desc += k
            pred = nms_preds[si] if si < len(nms_preds) else torch.zeros((0, 6), device=self.device)
            if int(pred.shape[0]) > 0 and self.strict_nms_conf > 0.0:
                keep = pred[:, 4] >= self.strict_nms_conf
                if bool(keep.any()):
                    pred = pred[keep]
                else:
                    pred = torch.zeros((0, 6), device=pred.device, dtype=pred.dtype)

            if int(pred.shape[0]) > 0:
                max_keep = max(k, gt_n * self.strict_max_candidates_factor)
                max_keep = max(1, min(int(pred.shape[0]), max_keep))
                if int(pred.shape[0]) > max_keep:
                    top_idx = torch.topk(pred[:, 4], k=max_keep, largest=True).indices
                    pred = pred[top_idx]

            m = int(pred.shape[0])
            if m <= 0:
                self.strict_miss += k
                continue

            self.strict_pred_desc += k
            cand_input = pred[:, :4]
            score = score_matrix_from_phrase_maps(
                phrase_logits_levels=self._strict_phrase_logits,
                image_index=si,
                token_indices=valid_tok,
                candidate_boxes_xyxy=cand_input,
                input_hw=input_hw,
            )

            row_ind, col_ind = assign_by_score_matrix(score, temperature=self.matching_temperature)
            assigned = {int(r): int(c) for r, c in zip(row_ind.tolist(), col_ind.tolist())}

            cand_native = pred[:, :4].clone()
            ops.scale_boxes(pbatch["imgsz"], cand_native, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
            iou_mat = box_iou_matrix(cand_native, gt_bbox) if int(gt_bbox.shape[0]) > 0 else torch.zeros((m, 0), device=self.device)

            for local_r, tok_idx_t in enumerate(valid_tok.tolist()):
                target = int(txt_target[si, tok_idx_t].item())
                cand = int(assigned.get(local_r, -1))
                if cand < 0 or cand >= m:
                    self.strict_miss += 1
                    continue

                if int(iou_mat.shape[1]) <= 0:
                    self.strict_hallucination += 1
                    continue

                assign_iou = float(iou_mat[cand].max().item()) if int(iou_mat.shape[1]) > 0 else 0.0
                target_iou = float(iou_mat[cand, target].item()) if 0 <= target < int(iou_mat.shape[1]) else 0.0
                best_gt = int(iou_mat[cand].argmax().item()) if int(iou_mat.shape[1]) > 0 else -1

                if target_iou >= self.strict_match_iou and best_gt == target:
                    self.strict_correct += 1
                elif assign_iou >= self.strict_match_iou and best_gt != target:
                    self.strict_misattribution += 1
                else:
                    self.strict_hallucination += 1

    def get_stats(self):
        if not self.strict_grounding:
            return super().get_stats()

        total = max(int(self.strict_total_desc), 1)
        pred = max(int(self.strict_pred_desc), 1)
        acc = float(self.strict_correct) / float(total)
        prec = float(self.strict_correct) / float(pred)
        rec = float(self.strict_correct) / float(total)

        return {
            "metrics/grounding_accuracy": acc,
            "metrics/grounding_precision": prec,
            "metrics/grounding_recall": rec,
            "metrics/misattribution_rate": float(self.strict_misattribution) / float(total),
            "metrics/hallucination_rate": float(self.strict_hallucination) / float(total),
            "metrics/miss_rate": float(self.strict_miss) / float(total),
            "metrics/desc_total": float(self.strict_total_desc),
            "metrics/desc_with_pred": float(self.strict_pred_desc),
        }

    def print_results(self):
        if not self.strict_grounding:
            return super().print_results()

        stats = self.get_stats()
        LOGGER.info(
            "Strict grounding: images=%d, desc=%d, correct=%d, misattr=%d, halluc=%d, miss=%d, acc=%.4f, prec=%.4f, rec=%.4f",
            int(self.seen),
            int(self.strict_total_desc),
            int(self.strict_correct),
            int(self.strict_misattribution),
            int(self.strict_hallucination),
            int(self.strict_miss),
            float(stats.get("metrics/grounding_accuracy", 0.0)),
            float(stats.get("metrics/grounding_precision", 0.0)),
            float(stats.get("metrics/grounding_recall", 0.0)),
        )
