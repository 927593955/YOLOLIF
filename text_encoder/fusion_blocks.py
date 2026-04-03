from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear + low-rank adaptation branch."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.rank = int(max(1, rank))
        self.alpha = float(max(1e-6, alpha))
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(max(0.0, dropout)))

        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


class LoRAConv2d(nn.Module):
    """Conv2d + low-rank 1x1 adaptation branch."""

    def __init__(self, base: nn.Conv2d, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.rank = int(max(1, rank))
        self.alpha = float(max(1e-6, alpha))
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout2d(float(max(0.0, dropout)))

        self.lora_a = nn.Conv2d(base.in_channels, self.rank, kernel_size=1, stride=1, padding=0, bias=False)
        self.lora_b = nn.Conv2d(self.rank, base.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


class FiLMBlock(nn.Module):
    """Feature-wise linear modulation from text alignment maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Linear(1, channels)
        self.bias = nn.Linear(1, channels)

    def forward(self, feat: torch.Tensor, align_logit: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        bsz, channels, _, _ = feat.shape
        pooled = align_logit.mean(dim=(2, 3))
        gamma = torch.tanh(self.scale(pooled)).view(bsz, channels, 1, 1)
        beta = self.bias(pooled).view(bsz, channels, 1, 1)
        return feat * (1.0 + strength * gamma) + strength * beta


class TextCrossAttentionBlock(nn.Module):
    """Visual queries attend to all text tokens with residual projection."""

    def __init__(
        self,
        in_channels: int,
        text_dim: int,
        attn_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.attn_dim = int(max(8, attn_dim))
        self.vis_in = nn.Conv2d(self.in_channels, self.attn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.txt_in = nn.Linear(int(text_dim), self.attn_dim, bias=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=int(max(1, num_heads)),
            dropout=float(max(0.0, dropout)),
            batch_first=True,
        )
        self.out = nn.Conv2d(self.attn_dim, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, feat: torch.Tensor, txt_vec: torch.Tensor, txt_token_mask: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, _, h, w = feat.shape
        vis_tokens = self.vis_in(feat).flatten(2).transpose(1, 2)
        txt_tokens = self.txt_in(txt_vec.to(device=feat.device, dtype=vis_tokens.dtype))

        key_padding_mask = None
        if isinstance(txt_token_mask, torch.Tensor) and txt_token_mask.ndim == 2 and int(txt_token_mask.shape[0]) == bsz:
            key_padding_mask = ~txt_token_mask.to(device=feat.device, dtype=torch.bool)

        attended, _ = self.attn(vis_tokens, txt_tokens, txt_tokens, key_padding_mask=key_padding_mask, need_weights=False)
        out = attended.transpose(1, 2).reshape(bsz, self.attn_dim, h, w)
        return feat + self.out(out)
