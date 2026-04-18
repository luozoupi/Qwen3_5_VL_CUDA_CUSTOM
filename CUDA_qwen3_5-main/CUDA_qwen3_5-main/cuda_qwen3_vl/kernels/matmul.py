from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from ._loader import load_op, maybe_strict_raise


def _ensure() -> bool:
    return load_op("matmul")


def _use_cublas_fallback() -> bool:
    """Env toggle: CUDA_QWEN3_VL_USE_CUBLAS=1 routes linear through PyTorch/cuBLAS."""
    return os.environ.get("CUDA_QWEN3_VL_USE_CUBLAS") == "1"


class _MatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.needs_bias_grad = bias is not None and bias.requires_grad
        if _use_cublas_fallback() or not _ensure():
            return F.linear(x, weight, bias)
        try:
            return torch.ops.cuda_qwen3_vl.matmul_forward(
                x.contiguous(), weight.contiguous(), bias.contiguous() if bias is not None else None
            )
        except Exception as exc:
            maybe_strict_raise("matmul", exc)
            return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        if _use_cublas_fallback() or not _ensure():
            # Fallback: standard linear backward
            gx = grad_output @ weight
            gw = grad_output.reshape(-1, grad_output.shape[-1]).T @ x.reshape(-1, x.shape[-1])
            gb = grad_output.reshape(-1, grad_output.shape[-1]).sum(0) if ctx.needs_bias_grad else None
            return gx, gw, gb
        try:
            outs = torch.ops.cuda_qwen3_vl.matmul_backward(
                x.contiguous(), weight.contiguous(), grad_output.contiguous(), ctx.needs_bias_grad
            )
            gx, gw = outs[0], outs[1]
            gb = outs[2] if ctx.needs_bias_grad else None
            # If bias was returned as empty tensor, treat as None
            if gb is not None and gb.numel() == 0:
                gb = None
            return gx, gw, gb
        except Exception as exc:
            maybe_strict_raise("matmul", exc)
            gx = grad_output @ weight
            gw = grad_output.reshape(-1, grad_output.shape[-1]).T @ x.reshape(-1, x.shape[-1])
            gb = grad_output.reshape(-1, grad_output.shape[-1]).sum(0) if ctx.needs_bias_grad else None
            return gx, gw, gb


def matmul(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    if not x.is_cuda:
        return F.linear(x, weight, bias)
    if _use_cublas_fallback():
        return F.linear(x, weight, bias)
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)):
        return _MatmulFunction.apply(x, weight, bias)
    if not _ensure():
        return F.linear(x, weight, bias)
    try:
        return torch.ops.cuda_qwen3_vl.matmul_forward(
            x.contiguous(), weight.contiguous(), bias.contiguous() if bias is not None else None
        )
    except Exception as exc:
        maybe_strict_raise("matmul", exc)
        return F.linear(x, weight, bias)
