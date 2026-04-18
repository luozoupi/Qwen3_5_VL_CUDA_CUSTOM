from __future__ import annotations

import torch
import torch.nn.functional as F

from ._loader import load_op, maybe_strict_raise


def _ensure() -> bool:
    return load_op("conv3d_patch")


def _fallback(x, weight, bias, stride):
    return F.conv3d(x, weight, bias=bias, stride=stride)


def conv3d_patch(
    x: torch.Tensor, weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Non-overlapping 3D patch embed where stride == kernel_size.

    Input x: (N, C, T, H, W) where (T, H, W) == kernel_size.
    Weight: (E, C, T, H, W). Output: (N, E).
    """
    if not x.is_cuda or not _ensure():
        # Fallback: use PyTorch conv3d. Requires stride matching.
        kernel_size = weight.shape[2:]
        return _fallback(x, weight, bias, stride=kernel_size).reshape(x.shape[0], -1)
    try:
        return torch.ops.cuda_qwen3_vl.conv3d_patch_forward(
            x.contiguous(), weight.contiguous(),
            bias.contiguous() if bias is not None else None
        )
    except Exception as exc:
        maybe_strict_raise("conv3d_patch", exc)
        kernel_size = weight.shape[2:]
        return _fallback(x, weight, bias, stride=kernel_size).reshape(x.shape[0], -1)
