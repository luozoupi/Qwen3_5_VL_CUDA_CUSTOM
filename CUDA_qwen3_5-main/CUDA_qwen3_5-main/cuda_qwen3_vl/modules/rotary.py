"""Rotary position embedding helpers for vision (2D) and text (MRoPE 3D)."""
from __future__ import annotations

import torch
from torch import nn

from cuda_qwen3_vl.kernels import apply_mrope, apply_rope


class Vision2DRoPE(nn.Module):
    """Vision rotary embedding: 2D position grid (H, W) with inv_freq cache.

    Produces cos/sin of shape (S, dim) for seqlen S = H*W tokens.
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs  # (S, dim//2)

    @staticmethod
    def apply(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to Q/K tensors shaped (B, H, S, D)."""
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


class TextMRoPE(nn.Module):
    """Text MRoPE: 3D positions (T, H, W) interleaved into the head dim via mrope_section."""

    def __init__(self, dim: int, theta: float = 500000.0, mrope_section: list[int] | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        # Default Qwen3-VL mrope_section [24, 20, 20] sums to half of rotary_dim (128 // 2 = 64)
        self.mrope_section = mrope_section or [24, 20, 20]
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def compute_cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin for 3D positions. position_ids: (3, B, S).

        Returns (cos_3d, sin_3d) of shape (3, B, S, dim).
        """
        # position_ids: (3, B, S), inv_freq: (dim//2,)
        # freqs: (3, B, S, dim//2)
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.to(position_ids.device)
        emb = torch.cat([freqs, freqs], dim=-1)  # (3, B, S, dim)
        return emb.cos(), emb.sin()

    def apply(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos_3d, sin_3d = self.compute_cos_sin(position_ids)
        return apply_mrope(q, cos_3d, sin_3d, self.mrope_section), apply_mrope(k, cos_3d, sin_3d, self.mrope_section)
