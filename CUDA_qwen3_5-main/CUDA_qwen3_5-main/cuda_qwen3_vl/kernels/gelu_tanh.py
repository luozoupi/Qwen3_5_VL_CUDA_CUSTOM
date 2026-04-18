from __future__ import annotations

import torch
import torch.nn.functional as F

from ._loader import load_op, maybe_strict_raise


def _ensure() -> bool:
    return load_op("gelu_tanh")


def _fallback(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


class _GeluTanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        if not _ensure():
            return _fallback(x)
        try:
            return torch.ops.cuda_qwen3_vl.gelu_tanh_forward(x.contiguous())
        except Exception as exc:
            maybe_strict_raise("gelu_tanh", exc)
            return _fallback(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        if not _ensure():
            with torch.enable_grad():
                x2 = x.detach().requires_grad_(True)
                y = _fallback(x2)
                (gx,) = torch.autograd.grad(y, x2, grad_output)
            return gx
        try:
            return torch.ops.cuda_qwen3_vl.gelu_tanh_backward(x.contiguous(), grad_output.contiguous())
        except Exception as exc:
            maybe_strict_raise("gelu_tanh", exc)
            with torch.enable_grad():
                x2 = x.detach().requires_grad_(True)
                y = _fallback(x2)
                (gx,) = torch.autograd.grad(y, x2, grad_output)
            return gx


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        return _fallback(x)
    if torch.is_grad_enabled() and x.requires_grad:
        return _GeluTanhFunction.apply(x)
    if not _ensure():
        return _fallback(x)
    try:
        return torch.ops.cuda_qwen3_vl.gelu_tanh_forward(x.contiguous())
    except Exception as exc:
        maybe_strict_raise("gelu_tanh", exc)
        return _fallback(x)
