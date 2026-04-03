from __future__ import annotations

from typing import Any, Dict


DEFAULT_TEXT_GUIDANCE_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "embedding_dir": "",
    "embedding_dim": 768,
    "phrase_types": ["NP", "PP", "ADJP"],
    "aggregation_mode": "lse",
    "phrase_type_weights": {
        "NP": 1.0,
        "PP": 1.2,
        "ADJP": 0.8,
        "FALLBACK": 1.0,
    },
    "guidance_strength": 0.25,
    "cls_gate_strength": 0.8,
    "cls_fusion_mode": "additive",
    "cls_gate_nonnegative": True,
    "cls_gate_temperature": 1.0,
    "cls_gate_bias_cap": 1.0,
    "alignment_temperature": 0.07,
    "fuse_temperature": 0.5,
    "lambda_heatmap": 0.3,
    "lambda_phrase": 0.2,
    "lambda_set": 0.6,
    "matching_temperature": 0.7,
    "matching_solver": "scipy_first",
    "strict_grounding": True,
    "strict_match_iou": 0.5,
    "strict_nms_conf": 0.25,
    "strict_max_candidates_factor": 4,
    "contrastive_loss_type": "logsigmoid_margin",
    "disable_augmentations": True,
    "text_seq_enhance": False,
    "text_seq_conv_layers": 1,
    "text_seq_kernel_size": 3,
    "text_seq_dropout": 0.0,
    "text_seq_pooling_mode": "none",
    "text_seq_pool_temperature": 1.0,
    "visual_attr_enabled": True,
    "visual_attr_include_geom": True,
    "visual_attr_include_stats": True,
    "visual_attr_scale": 1.0,
    "visual_attr_eps": 1e-6,
    "multi_proj_enabled": True,
    "multi_proj_score_scale": 1.0,
    "orth_loss_weight": 0.05,
    "lora_enabled": False,
    "lora_rank": 8,
    "lora_alpha": 16.0,
    "lora_dropout": 0.0,
    "film_enabled": True,
    "film_strength": 0.25,
    "cross_attn_enabled": True,
    "cross_attn_heads": 4,
    "cross_attn_dim": 128,
    "cross_attn_dropout": 0.0,
    "infonce_temperature": 0.25,
    "hard_neg_k": 32,
    "use_in_batch_negatives": True,
    "lambda_diou": 0.15,
    "diou_temperature": 1.0,
}

_TEXT_GUIDANCE_CONFIG: Dict[str, Any] = dict(DEFAULT_TEXT_GUIDANCE_CONFIG)


def configure_text_guidance(config: Dict[str, Any] | None = None) -> None:
    """Set process-level text guidance configuration for custom trainer/model."""
    global _TEXT_GUIDANCE_CONFIG
    _TEXT_GUIDANCE_CONFIG = dict(DEFAULT_TEXT_GUIDANCE_CONFIG)
    if config:
        _TEXT_GUIDANCE_CONFIG.update(config)


def get_text_guidance_config() -> Dict[str, Any]:
    """Return a copy of current text guidance settings."""
    return dict(_TEXT_GUIDANCE_CONFIG)
