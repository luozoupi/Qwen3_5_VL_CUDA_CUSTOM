"""Qwen3-VL vision transformer block: pre-LN + attention + residual + LN + MLP + residual."""
from __future__ import annotations

import torch
from torch import nn

from cuda_qwen3_vl.kernels import residual_add
from cuda_qwen3_vl.modules.attention import CudaVisionAttention
from cuda_qwen3_vl.modules.mlp import CudaVisionMLP
from cuda_qwen3_vl.modules.norms import CudaLayerNorm


class CudaVisionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = CudaLayerNorm(hidden_size, eps=eps, bias=True)
        self.attn = CudaVisionAttention(hidden_size, num_heads)
        self.norm2 = CudaLayerNorm(hidden_size, eps=eps, bias=True)
        self.mlp = CudaVisionMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = residual_add(self.attn(self.norm1(x), cos, sin), x)
        x = residual_add(self.mlp(self.norm2(x)), x)
        return x
