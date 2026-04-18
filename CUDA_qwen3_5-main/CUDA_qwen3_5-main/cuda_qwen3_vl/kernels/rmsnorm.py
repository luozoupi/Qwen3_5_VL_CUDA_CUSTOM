from __future__ import annotations

import torch

from ._loader import load_op, maybe_strict_raise


def _ensure() -> bool:
    return load_op("rmsnorm")


def _fallback_fwd(x, weight, eps):
    dtype = x.dtype
    x_fp = x.float()
    var = x_fp.pow(2).mean(dim=-1, keepdim=True)
    y = x_fp * torch.rsqrt(var + eps) * weight.float()
    return y.to(dtype)


def _fallback_bwd(x, weight, grad_out, eps):
    dtype = x.dtype
    x_fp = x.float()
    w_fp = weight.float()
    g_fp = grad_out.float()
    var = x_fp.pow(2).mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    normalized = x_fp * inv_std
    grad_weight = (g_fp * normalized).sum(dim=tuple(range(g_fp.ndim - 1)))
    dot = (g_fp * w_fp * x_fp).sum(dim=-1, keepdim=True)
    n_cols = x_fp.shape[-1]
    gx = g_fp * w_fp * inv_std - x_fp * (inv_std ** 3 / n_cols) * dot
    return gx.to(dtype), grad_weight.to(weight.dtype)


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        if not _ensure():
            return _fallback_fwd(x, weight, eps)
        try:
            return torch.ops.cuda_qwen3_vl.rmsnorm_forward(x.contiguous(), weight.contiguous(), eps)
        except Exception as exc:
            maybe_strict_raise("rmsnorm", exc)
            return _fallback_fwd(x, weight, eps)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        if not _ensure():
            gx, gw = _fallback_bwd(x, weight, grad_output, eps)
            return gx, gw, None
        try:
            gx, gw = torch.ops.cuda_qwen3_vl.rmsnorm_backward(
                x.contiguous(), weight.contiguous(), grad_output.contiguous(), eps
            )
            return gx, gw, None
        except Exception as exc:
            maybe_strict_raise("rmsnorm", exc)
            gx, gw = _fallback_bwd(x, weight, grad_output, eps)
            return gx, gw, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not x.is_cuda:
        return _fallback_fwd(x, weight, eps)
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return _RMSNormFunction.apply(x, weight, eps)
    if not _ensure():
        return _fallback_fwd(x, weight, eps)
    try:
        return torch.ops.cuda_qwen3_vl.rmsnorm_forward(x.contiguous(), weight.contiguous(), eps)
    except Exception as exc:
        maybe_strict_raise("rmsnorm", exc)
        return _fallback_fwd(x, weight, eps)
