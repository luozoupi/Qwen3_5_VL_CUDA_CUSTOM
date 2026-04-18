from __future__ import annotations

import torch

from ._loader import load_op, maybe_strict_raise


def _ensure() -> bool:
    return load_op("residual_add")


class _ResidualAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not _ensure():
            return a + b
        try:
            return torch.ops.cuda_qwen3_vl.residual_add_forward(a.contiguous(), b.contiguous())
        except Exception as exc:
            maybe_strict_raise("residual_add", exc)
            return a + b

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return grad_output, grad_output


def residual_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda:
        return a + b
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        return _ResidualAddFunction.apply(a, b)
    if not _ensure():
        return a + b
    try:
        return torch.ops.cuda_qwen3_vl.residual_add_forward(a.contiguous(), b.contiguous())
    except Exception as exc:
        maybe_strict_raise("residual_add", exc)
        return a + b
