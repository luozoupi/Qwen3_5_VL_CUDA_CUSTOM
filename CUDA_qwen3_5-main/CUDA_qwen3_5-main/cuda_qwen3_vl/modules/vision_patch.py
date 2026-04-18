"""Vision patch embed + positional embedding + merger for Qwen3-VL.

All inner compute paths are our custom CUDA kernels. Position-embed interpolation
and the grid-based 2D rotary are Python orchestration on top of torch ops (not
hot kernels — small-tensor bookkeeping).
"""
from __future__ import annotations

import torch
from torch import nn

from cuda_qwen3_vl.kernels import conv3d_patch, gelu_tanh
from cuda_qwen3_vl.modules.embedding import CudaEmbedding
from cuda_qwen3_vl.modules.linear import CudaLinear
from cuda_qwen3_vl.modules.norms import CudaLayerNorm


class CudaVisionPatchEmbed(nn.Module):
    """Qwen3-VL patch embed. Accepts HF's input layout: (seq_len, C*T*H*W) flat
    OR (N, C, T, H, W). Output: (seq_len, embed_dim).
    """

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int, temporal_patch_size: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        kernel_shape = (temporal_patch_size, patch_size, patch_size)
        self.weight = nn.Parameter(torch.empty(hidden_size, in_channels, *kernel_shape))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HF passes (seq_len, C*T*H*W) directly; reshape to 5D then run conv.
        if x.dim() == 2:
            x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return conv3d_patch(x, self.weight, self.bias)


class CudaVisionPatchMerger(nn.Module):
    """Merger with spatial-merge reshape.

    `use_postshuffle_norm` toggle decides whether norm runs before or after the
    merge-size reshape — matches HF's Qwen3VLVisionPatchMerger.
    """

    def __init__(self, hidden_size: int, out_hidden_size: int, spatial_merge_size: int = 2, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        self.hidden_size_in = hidden_size
        self.merged_hidden = hidden_size * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.merged_hidden if use_postshuffle_norm else hidden_size
        self.norm = CudaLayerNorm(norm_dim, eps=1e-6, bias=True)
        self.linear_fc1 = CudaLinear(self.merged_hidden, self.merged_hidden, bias=True)
        self.linear_fc2 = CudaLinear(self.merged_hidden, out_hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HF layout: norm over the per-token hidden, then reshape to merged.
        # For postshuffle variant, reshape first, then norm over merged hidden.
        if self.use_postshuffle_norm:
            x = x.view(-1, self.merged_hidden)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x.view(-1, self.merged_hidden)
        return self.linear_fc2(gelu_tanh(self.linear_fc1(x)))
