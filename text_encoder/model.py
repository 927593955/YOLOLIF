from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel

from .fusion_blocks import FiLMBlock, LoRAConv2d, LoRALinear, TextCrossAttentionBlock
from .losses import TextGuidedDetectionLoss


class TextGuidedDetectionModel(DetectionModel):
    """Detection model with phrase-level text alignment guidance."""

    def __init__(
        self,
        cfg="yolov8n.yaml",
        ch=3,
        nc=None,
        verbose=True,
        text_enabled: bool = True,
        text_embed_dim: int = 768,
        text_guidance_strength: float = 0.25,
        text_cls_gate_strength: float = 0.8,
        text_cls_fusion_mode: str = "additive",
        text_cls_gate_nonnegative: bool = True,
        text_cls_gate_temperature: float = 1.0,
        text_cls_gate_bias_cap: float = 1.0,
        text_alignment_temperature: float = 0.07,
        text_fuse_temperature: float = 0.5,
        text_aggregation_mode: str = "lse",
        text_seq_enhance: bool = False,
        text_seq_conv_layers: int = 1,
        text_seq_kernel_size: int = 3,
        text_seq_dropout: float = 0.0,
        text_seq_pooling_mode: str = "none",
        text_seq_pool_temperature: float = 1.0,
        visual_attr_enabled: bool = False,
        visual_attr_include_geom: bool = True,
        visual_attr_include_stats: bool = True,
        visual_attr_scale: float = 1.0,
        visual_attr_eps: float = 1e-6,
        multi_proj_enabled: bool = False,
        multi_proj_score_scale: float = 1.0,
        lora_enabled: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        film_enabled: bool = False,
        film_strength: float = 0.25,
        cross_attn_enabled: bool = False,
        cross_attn_heads: int = 4,
        cross_attn_dim: int = 128,
        cross_attn_dropout: float = 0.0,
    ):
        self.text_enabled = bool(text_enabled)
        self.text_embed_dim = int(text_embed_dim)
        self.text_guidance_strength = float(text_guidance_strength)
        self.text_cls_gate_strength = float(max(0.0, text_cls_gate_strength))
        fusion_mode = str(text_cls_fusion_mode or "additive").strip().lower()
        if fusion_mode in {"mul", "multiply", "multiplicative"}:
            fusion_mode = "multiplicative"
        elif fusion_mode in {"add", "additive"}:
            fusion_mode = "additive"
        else:
            fusion_mode = "additive"
        self.text_cls_fusion_mode = fusion_mode
        self.text_cls_gate_nonnegative = bool(text_cls_gate_nonnegative)
        self.text_cls_gate_temperature = float(max(1e-3, text_cls_gate_temperature))
        self.text_cls_gate_bias_cap = float(max(0.0, text_cls_gate_bias_cap))
        self.text_alignment_temperature = float(max(1e-3, text_alignment_temperature))
        self.text_fuse_temperature = float(max(1e-3, text_fuse_temperature))
        agg_mode = str(text_aggregation_mode or "lse").strip().lower()
        self.text_aggregation_mode = agg_mode if agg_mode in {"weighted_sum", "mean", "lse"} else "lse"

        self.text_seq_enhance = bool(text_seq_enhance)
        self.text_seq_conv_layers = int(max(1, text_seq_conv_layers))
        self.text_seq_kernel_size = int(max(1, text_seq_kernel_size))
        self.text_seq_dropout = float(max(0.0, text_seq_dropout))
        pool_mode = str(text_seq_pooling_mode or "none").strip().lower()
        self.text_seq_pooling_mode = pool_mode if pool_mode in {"none", "learnable_weight"} else "none"
        self.text_seq_pool_temperature = float(max(1e-3, text_seq_pool_temperature))

        self.visual_attr_enabled = bool(visual_attr_enabled)
        self.visual_attr_include_geom = bool(visual_attr_include_geom)
        self.visual_attr_include_stats = bool(visual_attr_include_stats)
        self.visual_attr_scale = float(visual_attr_scale)
        self.visual_attr_eps = float(max(1e-9, visual_attr_eps))
        self.multi_proj_enabled = bool(multi_proj_enabled)
        self.multi_proj_score_scale = float(max(0.0, multi_proj_score_scale))
        self.lora_enabled = bool(lora_enabled)
        self.lora_rank = int(max(1, lora_rank))
        self.lora_alpha = float(max(1e-6, lora_alpha))
        self.lora_dropout = float(max(0.0, lora_dropout))
        self.film_enabled = bool(film_enabled)
        self.film_strength = float(max(0.0, film_strength))
        self.cross_attn_enabled = bool(cross_attn_enabled)
        self.cross_attn_heads = int(max(1, cross_attn_heads))
        self.cross_attn_dim = int(max(8, cross_attn_dim))
        self.cross_attn_dropout = float(max(0.0, cross_attn_dropout))
        self.visual_attr_dim = 0
        if self.visual_attr_include_geom:
            # x, y, relative area
            self.visual_attr_dim += 3
        if self.visual_attr_include_stats:
            # z-score and centered activation
            self.visual_attr_dim += 2

        self._last_text_outputs: Dict = {}
        # Force nc=1 for class-agnostic text-guided detection
        super().__init__(cfg=cfg, ch=ch, nc=1, verbose=verbose)

        self.token_proj = nn.ModuleList()
        self.visual_proj = nn.ModuleList()
        self.visual_attr_proj = nn.ModuleList()
        self.token_proj_geo = nn.ModuleList()
        self.token_proj_attr = nn.ModuleList()
        self.token_proj_sem = nn.ModuleList()
        self.visual_proj_geo = nn.ModuleList()
        self.visual_proj_attr = nn.ModuleList()
        self.visual_proj_sem = nn.ModuleList()
        self.film_blocks = nn.ModuleList()
        self.cross_attn_blocks = nn.ModuleList()

        self.text_seq_blocks = nn.ModuleList()
        self.text_seq_norms = nn.ModuleList()
        self.text_seq_score = None

        if self.text_seq_enhance:
            k = self.text_seq_kernel_size
            pad = k // 2
            for _ in range(self.text_seq_conv_layers):
                block = nn.Sequential(
                    nn.Conv1d(self.text_embed_dim, self.text_embed_dim, kernel_size=k, padding=pad, groups=self.text_embed_dim, bias=False),
                    nn.SiLU(inplace=True),
                    nn.Conv1d(self.text_embed_dim, self.text_embed_dim, kernel_size=1, bias=False),
                    nn.Dropout(self.text_seq_dropout),
                )
                self.text_seq_blocks.append(block)
                self.text_seq_norms.append(nn.LayerNorm(self.text_embed_dim))

            if self.text_seq_pooling_mode == "learnable_weight":
                self.text_seq_score = nn.Linear(self.text_embed_dim, 1, bias=False)

        self._build_text_fusion_layers()

        self.text_lambda_heatmap = 0.3
        self.text_lambda_phrase = 0.2

    def _infer_head_input_channels(self, head: Detect, idx: int) -> int:
        branch = head.cv2[idx]
        if isinstance(branch, nn.Sequential) and len(branch) > 0:
            first = branch[0]
            if hasattr(first, "conv") and hasattr(first.conv, "in_channels"):
                return int(first.conv.in_channels)
        raise RuntimeError("Cannot infer Detect head input channels for text guidance")

    def _build_text_fusion_layers(self) -> None:
        head = self.model[-1]
        if not isinstance(head, Detect):
            return

        for i in range(head.nl):
            in_ch = self._infer_head_input_channels(head, i)
            proj_dim = max(32, min(256, in_ch, self.text_embed_dim))
            token_proj = nn.Linear(self.text_embed_dim, proj_dim, bias=False)
            visual_proj = nn.Conv2d(in_ch, proj_dim, kernel_size=1, stride=1, padding=0, bias=False)
            if self.lora_enabled:
                token_proj = LoRALinear(token_proj, rank=self.lora_rank, alpha=self.lora_alpha, dropout=self.lora_dropout)
                visual_proj = LoRAConv2d(visual_proj, rank=self.lora_rank, alpha=self.lora_alpha, dropout=self.lora_dropout)
            self.token_proj.append(token_proj)
            self.visual_proj.append(visual_proj)
            self.film_blocks.append(FiLMBlock(in_ch))
            self.cross_attn_blocks.append(
                TextCrossAttentionBlock(
                    in_channels=in_ch,
                    text_dim=self.text_embed_dim,
                    attn_dim=self.cross_attn_dim,
                    num_heads=self.cross_attn_heads,
                    dropout=self.cross_attn_dropout,
                )
            )
            if self.multi_proj_enabled:
                token_proj_geo = nn.Linear(self.text_embed_dim, proj_dim, bias=False)
                token_proj_attr = nn.Linear(self.text_embed_dim, proj_dim, bias=False)
                token_proj_sem = nn.Linear(self.text_embed_dim, proj_dim, bias=False)
                visual_proj_geo = nn.Conv2d(in_ch, proj_dim, kernel_size=1, stride=1, padding=0, bias=False)
                visual_proj_attr = nn.Conv2d(in_ch, proj_dim, kernel_size=1, stride=1, padding=0, bias=False)
                visual_proj_sem = nn.Conv2d(in_ch, proj_dim, kernel_size=1, stride=1, padding=0, bias=False)
                if self.lora_enabled:
                    token_proj_geo = LoRALinear(token_proj_geo, rank=self.lora_rank, alpha=self.lora_alpha, dropout=self.lora_dropout)
                    token_proj_attr = LoRALinear(token_proj_attr, rank=self.lora_rank, alpha=self.lora_alpha, dropout=self.lora_dropout)
                    token_proj_sem = LoRALinear(token_proj_sem, rank=self.lora_rank, alpha=self.lora_alpha, dropout=self.lora_dropout)
                    visual_proj_geo = LoRAConv2d(visual_proj_geo, rank=self.lora_rank, alpha=self.lora_alpha, dropout=self.lora_dropout)
                    visual_proj_attr = LoRAConv2d(visual_proj_attr, rank=self.lora_rank, alpha=self.lora_alpha, dropout=self.lora_dropout)
                    visual_proj_sem = LoRAConv2d(visual_proj_sem, rank=self.lora_rank, alpha=self.lora_alpha, dropout=self.lora_dropout)
                self.token_proj_geo.append(token_proj_geo)
                self.token_proj_attr.append(token_proj_attr)
                self.token_proj_sem.append(token_proj_sem)
                self.visual_proj_geo.append(visual_proj_geo)
                self.visual_proj_attr.append(visual_proj_attr)
                self.visual_proj_sem.append(visual_proj_sem)
            if self.visual_attr_enabled and self.visual_attr_dim > 0:
                self.visual_attr_proj.append(
                    nn.Conv2d(self.visual_attr_dim, proj_dim, kernel_size=1, stride=1, padding=0, bias=False)
                )

    def _apply_cross_attention(
        self,
        feats: List[torch.Tensor],
        txt_vec: torch.Tensor,
        txt_token_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        if not self.cross_attn_enabled:
            return feats

        out: List[torch.Tensor] = []
        for i, feat in enumerate(feats):
            if i >= len(self.cross_attn_blocks):
                out.append(feat)
                continue
            out.append(self.cross_attn_blocks[i](feat, txt_vec, txt_token_mask))
        return out

    def _enhance_text_sequence(
        self,
        txt_vec: torch.Tensor,
        txt_token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.text_seq_enhance or len(self.text_seq_blocks) == 0:
            return txt_vec, None

        mask = txt_token_mask.to(device=txt_vec.device).unsqueeze(-1).to(dtype=txt_vec.dtype)
        x = txt_vec * mask

        for block, norm in zip(self.text_seq_blocks, self.text_seq_norms):
            y = block(x.transpose(1, 2)).transpose(1, 2)
            y = y * mask
            x = norm(x + y)
            x = x * mask

        adaptive_weight = None
        if self.text_seq_pooling_mode == "learnable_weight" and isinstance(self.text_seq_score, nn.Module):
            score = self.text_seq_score(x).squeeze(-1)
            neg_inf = torch.finfo(score.dtype).min
            score = torch.where(txt_token_mask, score / self.text_seq_pool_temperature, torch.full_like(score, neg_inf))
            adaptive_weight = torch.softmax(score, dim=1)
            valid_text = txt_token_mask.any(dim=1, keepdim=True)
            adaptive_weight = torch.where(valid_text, adaptive_weight, torch.zeros_like(adaptive_weight))

        return x, adaptive_weight

    def _build_visual_attributes(self, feat: torch.Tensor) -> Optional[torch.Tensor]:
        if (not self.visual_attr_enabled) or self.visual_attr_dim <= 0:
            return None

        bsz, _, h, w = feat.shape
        attrs: List[torch.Tensor] = []
        if self.visual_attr_include_geom:
            yy = torch.linspace(-1.0, 1.0, steps=h, device=feat.device, dtype=feat.dtype).view(1, 1, h, 1)
            xx = torch.linspace(-1.0, 1.0, steps=w, device=feat.device, dtype=feat.dtype).view(1, 1, 1, w)
            y_map = yy.expand(bsz, 1, h, w)
            x_map = xx.expand(bsz, 1, h, w)
            area = torch.full((bsz, 1, h, w), 1.0 / max(1.0, float(h * w)), device=feat.device, dtype=feat.dtype)
            attrs.extend([x_map, y_map, area])

        if self.visual_attr_include_stats:
            act = feat.mean(dim=1, keepdim=True)
            mean = act.mean(dim=(2, 3), keepdim=True)
            std = act.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(self.visual_attr_eps)
            z = (act - mean) / std
            centered = act - mean
            attrs.extend([z, centered])

        if not attrs:
            return None
        return torch.cat(attrs, dim=1) * self.visual_attr_scale

    @staticmethod
    def _normalize_text_inputs(
        txt_vec: torch.Tensor,
        txt_token_mask: Optional[torch.Tensor],
        txt_phrase_weight: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if txt_vec.ndim == 2:
            txt_vec = txt_vec.unsqueeze(1)
        if txt_vec.ndim != 3:
            raise ValueError(f"Expected txt_vec as [B, T, D] or [B, D], got {tuple(txt_vec.shape)}")

        bsz, tokens, _ = txt_vec.shape
        if isinstance(txt_token_mask, torch.Tensor) and txt_token_mask.ndim == 2 and txt_token_mask.shape[0] == bsz:
            token_mask = txt_token_mask.to(device=txt_vec.device, dtype=torch.bool)
            if int(token_mask.shape[1]) != tokens:
                if int(token_mask.shape[1]) > tokens:
                    token_mask = token_mask[:, :tokens]
                else:
                    pad = torch.zeros((bsz, tokens - int(token_mask.shape[1])), device=txt_vec.device, dtype=torch.bool)
                    token_mask = torch.cat((token_mask, pad), dim=1)
        else:
            token_mask = torch.ones((bsz, tokens), device=txt_vec.device, dtype=torch.bool)

        if not txt_vec.is_floating_point():
            txt_vec = txt_vec.float()

        if isinstance(txt_phrase_weight, torch.Tensor) and txt_phrase_weight.ndim == 2 and txt_phrase_weight.shape[0] == bsz:
            phrase_weight = txt_phrase_weight.to(device=txt_vec.device, dtype=txt_vec.dtype)
            if int(phrase_weight.shape[1]) != tokens:
                if int(phrase_weight.shape[1]) > tokens:
                    phrase_weight = phrase_weight[:, :tokens]
                else:
                    pad = torch.zeros((bsz, tokens - int(phrase_weight.shape[1])), device=txt_vec.device, dtype=txt_vec.dtype)
                    phrase_weight = torch.cat((phrase_weight, pad), dim=1)
        else:
            phrase_weight = torch.ones((bsz, tokens), device=txt_vec.device, dtype=txt_vec.dtype)

        phrase_weight = phrase_weight.clamp_min(0.0) * token_mask.to(dtype=txt_vec.dtype)

        return txt_vec, token_mask, phrase_weight

    @staticmethod
    def _normalize_phrase_weights(phrase_weight: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        mask_f = token_mask.to(dtype=phrase_weight.dtype)
        weights = phrase_weight * mask_f
        denom = weights.sum(dim=1, keepdim=True)

        empty = denom.squeeze(1) <= 0
        if empty.any():
            fallback = mask_f[empty]
            fallback_sum = fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
            weights = weights.clone()
            weights[empty] = fallback / fallback_sum
            denom = weights.sum(dim=1, keepdim=True)

        return weights / denom.clamp_min(1e-6)

    @staticmethod
    def _module_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
        first = next(module.parameters(), None)
        return first.dtype if isinstance(first, torch.Tensor) else fallback

    def _compute_alignment_logits(
        self,
        feats: List[torch.Tensor],
        txt_vec: torch.Tensor,
        txt_token_mask: torch.Tensor,
        txt_phrase_weight: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        fused_logits: List[torch.Tensor] = []
        phrase_logits: List[torch.Tensor] = []
        branch_visual_feats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        valid_text = txt_token_mask.any(dim=1)

        for i, feat in enumerate(feats):
            if not isinstance(feat, torch.Tensor):
                continue
            if i >= len(self.token_proj) or i >= len(self.visual_proj):
                fused_logits.append(torch.zeros((feat.shape[0], 1, feat.shape[2], feat.shape[3]), device=feat.device, dtype=feat.dtype))
                phrase_logits.append(
                    torch.zeros((feat.shape[0], txt_vec.shape[1], feat.shape[2], feat.shape[3]), device=feat.device, dtype=feat.dtype)
                )
                continue

            bsz, _, h, w = feat.shape
            vis_attr = None
            if self.visual_attr_enabled and i < len(self.visual_attr_proj):
                vis_attr = self._build_visual_attributes(feat)

            if self.multi_proj_enabled and i < len(self.token_proj_geo) and i < len(self.visual_proj_geo):
                txt_input_geo = txt_vec.to(device=feat.device, dtype=self._module_dtype(self.token_proj_geo[i], feat.dtype))
                txt_input_attr = txt_vec.to(device=feat.device, dtype=self._module_dtype(self.token_proj_attr[i], feat.dtype))
                txt_input_sem = txt_vec.to(device=feat.device, dtype=self._module_dtype(self.token_proj_sem[i], feat.dtype))

                txt_geo = F.normalize(self.token_proj_geo[i](txt_input_geo).to(dtype=feat.dtype), dim=-1)
                txt_attr = F.normalize(self.token_proj_attr[i](txt_input_attr).to(dtype=feat.dtype), dim=-1)
                txt_sem = F.normalize(self.token_proj_sem[i](txt_input_sem).to(dtype=feat.dtype), dim=-1)

                vis_geo = self.visual_proj_geo[i](feat)
                vis_attr_map = self.visual_proj_attr[i](feat)
                vis_sem = self.visual_proj_sem[i](feat)
                if isinstance(vis_attr, torch.Tensor):
                    attr_proj = self.visual_attr_proj[i](vis_attr.to(dtype=feat.dtype))
                    vis_geo = vis_geo + attr_proj
                    vis_attr_map = vis_attr_map + attr_proj
                    vis_sem = vis_sem + attr_proj

                vis_geo_tokens = F.normalize(vis_geo.flatten(2).transpose(1, 2), dim=-1)
                vis_attr_tokens = F.normalize(vis_attr_map.flatten(2).transpose(1, 2), dim=-1)
                vis_sem_tokens = F.normalize(vis_sem.flatten(2).transpose(1, 2), dim=-1)

                sim_geo = torch.einsum("btp,bnp->btn", txt_geo, vis_geo_tokens)
                sim_attr = torch.einsum("btp,bnp->btn", txt_attr, vis_attr_tokens)
                sim_sem = torch.einsum("btp,bnp->btn", txt_sem, vis_sem_tokens)
                sim = (sim_geo + sim_attr + sim_sem) * self.multi_proj_score_scale / self.text_alignment_temperature
                branch_visual_feats.append((vis_geo, vis_attr_map, vis_sem))
            else:
                token_proj = self.token_proj[i]
                txt_input = txt_vec.to(device=feat.device, dtype=self._module_dtype(token_proj, feat.dtype))
                txt_proj = token_proj(txt_input).to(dtype=feat.dtype)
                txt_proj = F.normalize(txt_proj, dim=-1)

                vis_proj = self.visual_proj[i](feat)
                if isinstance(vis_attr, torch.Tensor):
                    vis_proj = vis_proj + self.visual_attr_proj[i](vis_attr.to(dtype=feat.dtype))
                vis_tokens = vis_proj.flatten(2).transpose(1, 2)
                vis_tokens = F.normalize(vis_tokens, dim=-1)

                sim = torch.einsum("btp,bnp->btn", txt_proj, vis_tokens) / self.text_alignment_temperature

            mask = txt_token_mask.to(device=sim.device)
            raw_weight = txt_phrase_weight.to(device=sim.device, dtype=sim.dtype)
            if self.text_aggregation_mode == "mean":
                agg_weight = self._normalize_phrase_weights(mask.to(dtype=sim.dtype), mask)
            else:
                agg_weight = self._normalize_phrase_weights(raw_weight, mask)

            valid_token = mask.unsqueeze(-1)
            # Remove token-wise spatial bias before phrase fusion to reduce global activation drift.
            sim_centered = sim - sim.mean(dim=-1, keepdim=True)
            phrase_map = torch.where(valid_token, sim_centered, torch.zeros_like(sim_centered))

            if self.text_aggregation_mode == "lse":
                lse_input = phrase_map / self.text_fuse_temperature
                lse_input = lse_input + torch.log(agg_weight.clamp_min(1e-6)).unsqueeze(-1)
                neg_inf = torch.finfo(lse_input.dtype).min
                lse_input = torch.where(valid_token, lse_input, torch.full_like(lse_input, neg_inf))
                fused = self.text_fuse_temperature * torch.logsumexp(lse_input, dim=1)
            else:
                fused = torch.einsum("btn,bt->bn", phrase_map, agg_weight)

            if (~valid_text).any():
                phrase_map = torch.where(valid_text.view(-1, 1, 1), phrase_map, torch.zeros_like(phrase_map))
                fused = torch.where(valid_text.view(-1, 1), fused, torch.zeros_like(fused))

            phrase_logits.append(phrase_map.view(bsz, int(phrase_map.shape[1]), h, w).to(dtype=feat.dtype))
            fused_logits.append(fused.view(bsz, 1, h, w).to(dtype=feat.dtype))

        return fused_logits, phrase_logits, branch_visual_feats

    def _apply_feature_guidance(self, feats: List[torch.Tensor], alignment_logits: List[torch.Tensor]) -> List[torch.Tensor]:
        guided: List[torch.Tensor] = []
        for i, feat in enumerate(feats):
            if i >= len(alignment_logits):
                guided.append(feat)
                continue
            gate = torch.sigmoid(alignment_logits[i]) - 0.5
            gated = feat * (1.0 + self.text_guidance_strength * gate)
            if self.film_enabled and i < len(self.film_blocks):
                guided.append(self.film_blocks[i](gated, alignment_logits[i], strength=self.film_strength))
            else:
                guided.append(gated)
        return guided

    def _apply_cls_logits_gating(self, raw_maps: List[torch.Tensor], alignment_logits: List[torch.Tensor], head: Detect) -> List[torch.Tensor]:
        cls_start = int(head.reg_max * 4)
        gated_maps: List[torch.Tensor] = []
        # Backward-compatible defaults for models loaded from older checkpoints.
        cls_gate_temperature = float(max(1e-3, getattr(self, "text_cls_gate_temperature", 1.0)))
        cls_gate_nonnegative = bool(getattr(self, "text_cls_gate_nonnegative", True))
        cls_gate_bias_cap = float(max(0.0, getattr(self, "text_cls_gate_bias_cap", 1.0)))
        cls_fusion_mode = str(getattr(self, "text_cls_fusion_mode", "multiplicative")).strip().lower()

        for i, level_map in enumerate(raw_maps):
            if not isinstance(level_map, torch.Tensor) or int(level_map.shape[1]) <= cls_start:
                gated_maps.append(level_map)
                continue

            if i < len(alignment_logits):
                gate = alignment_logits[i]
                if tuple(gate.shape[-2:]) != tuple(level_map.shape[-2:]):
                    gate = F.interpolate(gate, size=level_map.shape[-2:], mode="bilinear", align_corners=False)
            else:
                gate = torch.zeros((level_map.shape[0], 1, level_map.shape[2], level_map.shape[3]), device=level_map.device, dtype=level_map.dtype)

            gate_signal = torch.sigmoid(gate / cls_gate_temperature).to(dtype=level_map.dtype)
            if cls_gate_nonnegative:
                cls_bias = gate_signal
            else:
                cls_bias = gate_signal - 0.5

            if cls_gate_bias_cap > 0.0:
                if cls_gate_nonnegative:
                    cls_bias = cls_bias.clamp(min=0.0, max=cls_gate_bias_cap)
                else:
                    cls_bias = cls_bias.clamp(min=-cls_gate_bias_cap, max=cls_gate_bias_cap)

            cls_logits_raw = level_map[:, cls_start:, :, :]
            if cls_fusion_mode == "multiplicative":
                gate_scale = 1.0 + self.text_cls_gate_strength * cls_bias
                cls_logits = cls_logits_raw * gate_scale
            else:
                cls_logits = cls_logits_raw + self.text_cls_gate_strength * cls_bias

            merged = torch.cat((level_map[:, :cls_start, :, :], cls_logits), dim=1)
            gated_maps.append(merged)

        return gated_maps

    def _apply_prediction_gating(self, pred, alignment_logits: List[torch.Tensor], head: Detect):
        if isinstance(pred, list):
            return self._apply_cls_logits_gating(pred, alignment_logits, head)

        if isinstance(pred, tuple) and len(pred) == 2 and isinstance(pred[1], list):
            gated_maps = self._apply_cls_logits_gating(pred[1], alignment_logits, head)
            decoded = head._inference(gated_maps)
            return decoded if head.export else (decoded, gated_maps)

        return pred

    def _predict_with_text(
        self,
        x: torch.Tensor,
        txt_vec: torch.Tensor,
        txt_token_mask: Optional[torch.Tensor] = None,
        txt_phrase_weight: Optional[torch.Tensor] = None,
    ):
        txt_vec = txt_vec.to(device=x.device)
        txt_vec, txt_token_mask, txt_phrase_weight = self._normalize_text_inputs(txt_vec, txt_token_mask, txt_phrase_weight)
        txt_vec, txt_adaptive_weight = self._enhance_text_sequence(txt_vec, txt_token_mask)

        effective_phrase_weight = txt_phrase_weight
        if isinstance(txt_adaptive_weight, torch.Tensor):
            effective_phrase_weight = txt_phrase_weight * txt_adaptive_weight.to(device=txt_phrase_weight.device, dtype=txt_phrase_weight.dtype)

        y = []
        for m in self.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)

        head = self.model[-1]
        if not isinstance(head, Detect):
            head_inputs = [y[j] for j in head.f] if isinstance(head.f, list) else [y[head.f]]
            pred = head(head_inputs)
            self._last_text_outputs = {
                "alignment_logits": [],
                "fused_logits": [],
                "alignment_maps": [],
                "phrase_logits": [],
                "branch_visual_feats": [],
                "text_token_mask": txt_token_mask,
                "text_phrase_weight": txt_phrase_weight,
            }
            return pred

        head_inputs = [y[j] for j in head.f]
        head_inputs = self._apply_cross_attention(head_inputs, txt_vec, txt_token_mask)
        alignment_logits, phrase_logits, branch_visual_feats = self._compute_alignment_logits(
            head_inputs,
            txt_vec,
            txt_token_mask,
            effective_phrase_weight,
        )
        fused_inputs = self._apply_feature_guidance(head_inputs, alignment_logits)
        pred = head(fused_inputs)
        pred = self._apply_prediction_gating(pred, alignment_logits, head)

        self._last_text_outputs = {
            "alignment_logits": alignment_logits,
            "fused_logits": alignment_logits,
            "alignment_maps": [torch.sigmoid(m) for m in alignment_logits],
            "phrase_logits": phrase_logits,
            "branch_visual_feats": branch_visual_feats,
            "text_token_mask": txt_token_mask,
            "text_phrase_weight": effective_phrase_weight,
        }
        return pred

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None, txt_vec=None, txt_token_mask=None, txt_phrase_weight=None):
        if augment or profile or visualize or embed:
            self._last_text_outputs = {}
            return super().predict(x, profile=profile, visualize=visualize, augment=augment, embed=embed)

        if not self.text_enabled or txt_vec is None:
            self._last_text_outputs = {}
            return super().predict(x, profile=profile, visualize=visualize, augment=False, embed=embed)

        return self._predict_with_text(x, txt_vec, txt_token_mask=txt_token_mask, txt_phrase_weight=txt_phrase_weight)

    def init_criterion(self):
        return TextGuidedDetectionLoss(self)

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            txt_vec = batch.get("txt_vec") if self.text_enabled else None
            txt_token_mask = batch.get("text_token_mask") if self.text_enabled else None
            txt_phrase_weight = batch.get("text_phrase_weight") if self.text_enabled else None
            preds = self.predict(
                batch["img"],
                txt_vec=txt_vec,
                txt_token_mask=txt_token_mask,
                txt_phrase_weight=txt_phrase_weight,
            )

        return self.criterion(preds, batch, text_outputs=self._last_text_outputs)
