from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import List

import torch

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel

from .dataset import TextGuidedYOLODataset
from .embedding_store import ObjectTextEmbeddingStore
from .model import TextGuidedDetectionModel
from .settings import get_text_guidance_config
from .validator import TextGuidedDetectionValidator


class TextGuidedDetectionTrainer(DetectionTrainer):
    """Detection trainer that injects image-level text embeddings into training batches."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        if cfg is None:
            super().__init__(overrides=overrides, _callbacks=_callbacks)
        else:
            super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        self.text_cfg = get_text_guidance_config()
        self.text_enabled = bool(self.text_cfg.get("enabled", False))
        self.embedding_store = None

        if self.text_enabled and bool(self.text_cfg.get("disable_augmentations", True)):
            self.args.mosaic = 0.0
            self.args.mixup = 0.0
            self.args.copy_paste = 0.0
            self.args.degrees = 0.0
            self.args.translate = 0.0
            self.args.scale = 0.0
            self.args.shear = 0.0
            self.args.perspective = 0.0
            self.args.flipud = 0.0
            self.args.fliplr = 0.0
            self.args.hsv_h = 0.0
            self.args.hsv_s = 0.0
            self.args.hsv_v = 0.0
            self.args.close_mosaic = 0
            self.args.multi_scale = False

        if self.text_enabled:
            embedding_root = self._resolve_embedding_root()
            self.embedding_store = ObjectTextEmbeddingStore(
                embedding_root=embedding_root,
                expected_dim=int(self.text_cfg.get("embedding_dim", 768)),
            )
            LOGGER.info(
                f"Text guidance enabled with {len(self.embedding_store.loaded_files)} embedding file(s), "
                f"dim={self.embedding_store.dim}"
            )

    def _resolve_embedding_root(self) -> str:
        configured = str(self.text_cfg.get("embedding_dir", "") or "").strip()
        if configured:
            return configured

        data_root = str(self.data.get("path", "") or "").strip()
        if not data_root:
            raise ValueError("Unable to resolve embedding root: set --text-embedding-dir explicitly.")
        return str(Path(data_root).expanduser().resolve())

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return TextGuidedYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = TextGuidedDetectionModel(
            cfg,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
            text_enabled=self.text_enabled,
            text_embed_dim=int(self.text_cfg.get("embedding_dim", 768)),
            text_guidance_strength=float(self.text_cfg.get("guidance_strength", 0.25)),
            text_cls_gate_strength=float(self.text_cfg.get("cls_gate_strength", 0.8)),
            text_cls_fusion_mode=str(self.text_cfg.get("cls_fusion_mode", "additive")),
            text_cls_gate_nonnegative=bool(self.text_cfg.get("cls_gate_nonnegative", True)),
            text_cls_gate_temperature=float(self.text_cfg.get("cls_gate_temperature", 1.0)),
            text_cls_gate_bias_cap=float(self.text_cfg.get("cls_gate_bias_cap", 1.0)),
            text_alignment_temperature=float(self.text_cfg.get("alignment_temperature", 0.07)),
            text_fuse_temperature=float(self.text_cfg.get("fuse_temperature", 0.5)),
            text_aggregation_mode=str(self.text_cfg.get("aggregation_mode", "weighted_sum")),
            text_seq_enhance=bool(self.text_cfg.get("text_seq_enhance", False)),
            text_seq_conv_layers=int(self.text_cfg.get("text_seq_conv_layers", 1)),
            text_seq_kernel_size=int(self.text_cfg.get("text_seq_kernel_size", 3)),
            text_seq_dropout=float(self.text_cfg.get("text_seq_dropout", 0.0)),
            text_seq_pooling_mode=str(self.text_cfg.get("text_seq_pooling_mode", "none")),
            text_seq_pool_temperature=float(self.text_cfg.get("text_seq_pool_temperature", 1.0)),
            visual_attr_enabled=bool(self.text_cfg.get("visual_attr_enabled", False)),
            visual_attr_include_geom=bool(self.text_cfg.get("visual_attr_include_geom", True)),
            visual_attr_include_stats=bool(self.text_cfg.get("visual_attr_include_stats", True)),
            visual_attr_scale=float(self.text_cfg.get("visual_attr_scale", 1.0)),
            visual_attr_eps=float(self.text_cfg.get("visual_attr_eps", 1e-6)),
            multi_proj_enabled=bool(self.text_cfg.get("multi_proj_enabled", False)),
            multi_proj_score_scale=float(self.text_cfg.get("multi_proj_score_scale", 1.0)),
            lora_enabled=bool(self.text_cfg.get("lora_enabled", False)),
            lora_rank=int(self.text_cfg.get("lora_rank", 8)),
            lora_alpha=float(self.text_cfg.get("lora_alpha", 16.0)),
            lora_dropout=float(self.text_cfg.get("lora_dropout", 0.0)),
            film_enabled=bool(self.text_cfg.get("film_enabled", False)),
            film_strength=float(self.text_cfg.get("film_strength", 0.25)),
            cross_attn_enabled=bool(self.text_cfg.get("cross_attn_enabled", False)),
            cross_attn_heads=int(self.text_cfg.get("cross_attn_heads", 4)),
            cross_attn_dim=int(self.text_cfg.get("cross_attn_dim", 128)),
            cross_attn_dropout=float(self.text_cfg.get("cross_attn_dropout", 0.0)),
        )
        model.text_lambda_heatmap = float(self.text_cfg.get("lambda_heatmap", 0.3))
        model.text_lambda_phrase = float(self.text_cfg.get("lambda_phrase", 0.2))
        model.text_lambda_set = float(self.text_cfg.get("lambda_set", 0.6))
        model.text_matching_temperature = float(self.text_cfg.get("matching_temperature", 0.7))
        model.text_orth_loss_weight = float(self.text_cfg.get("orth_loss_weight", 0.05))
        model.text_contrastive_loss_type = str(self.text_cfg.get("contrastive_loss_type", "logsigmoid_margin"))
        model.text_infonce_temperature = float(self.text_cfg.get("infonce_temperature", 0.25))
        model.text_hard_neg_k = int(self.text_cfg.get("hard_neg_k", 32))
        model.text_use_in_batch_negatives = bool(self.text_cfg.get("use_in_batch_negatives", True))
        model.text_lambda_diou = float(self.text_cfg.get("lambda_diou", 0.15))
        model.text_diou_temperature = float(self.text_cfg.get("diou_temperature", 1.0))

        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        # Keep names concise so terminal headers and csv columns stay readable.
        self.loss_names = "box", "cls", "dfl", "hm", "phr", "set", "ort", "diou", "tot"
        return TextGuidedDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            embedding_store=self.embedding_store,
        )

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

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if not self.text_enabled or self.embedding_store is None:
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

        # Force bounding box classes to 0 for class-agnostic YOLO detection head
        if "cls" in batch and batch["cls"] is not None:
            batch["cls"] = torch.zeros_like(batch["cls"])

        return batch
