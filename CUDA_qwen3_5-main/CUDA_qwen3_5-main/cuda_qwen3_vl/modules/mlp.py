from __future__ import annotations

import torch
from torch import nn

from cuda_qwen3_vl.kernels import gelu_tanh, swiglu
from cuda_qwen3_vl.modules.linear import CudaLinear


class CudaSwiGLUMLP(nn.Module):
    """Qwen3-VL text MLP: swiglu(gate, up) via custom CUDA kernel + CudaLinear."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = CudaLinear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = CudaLinear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = CudaLinear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(swiglu(gate, up))


class CudaVisionMLP(nn.Module):
    """Qwen3-VL vision MLP: linear_fc1 -> gelu_tanh -> linear_fc2. Bias=True per HF."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.linear_fc1 = CudaLinear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = CudaLinear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(gelu_tanh(self.linear_fc1(x)))
