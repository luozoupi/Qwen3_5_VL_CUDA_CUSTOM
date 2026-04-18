from __future__ import annotations

import torch
from torch import nn

from cuda_qwen3_vl.kernels import layernorm, rmsnorm


class CudaRMSNorm(nn.Module):
    """RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}"


class CudaLayerNorm(nn.Module):
    """LayerNorm with custom CUDA kernel. Matches nn.LayerNorm parameter layout."""

    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = True) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layernorm(x, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape[0]}, eps={self.eps}, bias={self.bias is not None}"
