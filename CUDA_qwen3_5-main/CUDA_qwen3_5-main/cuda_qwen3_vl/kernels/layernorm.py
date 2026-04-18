from __future__ import annotations

import torch
import torch.nn.functional as F

from ._loader import load_op, maybe_strict_raise


def _ensure() -> bool:
    return load_op("layernorm")


def _fallback(x, weight, bias, eps):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


class _LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.save_for_backward(x, weight, bias if bias is not None else torch.tensor([]))
        ctx.eps = eps
        ctx.has_bias = bias is not None
        if not _ensure():
            return _fallback(x, weight, bias, eps)
        try:
            return torch.ops.cuda_qwen3_vl.layernorm_forward(
                x.contiguous(), weight.contiguous(),
                bias.contiguous() if bias is not None else None, eps
            )
        except Exception as exc:
            maybe_strict_raise("layernorm", exc)
            return _fallback(x, weight, bias, eps)

    @staticmethod
    def backward(ctx, grad_output):
        # Use PyTorch autograd for backward — LayerNorm bwd has mean+variance corrections
        # that are non-trivial; reference LN backward is fast and correct.
        saved = ctx.saved_tensors
        x, weight = saved[0], saved[1]
        bias = saved[2] if ctx.has_bias and saved[2].numel() > 0 else None
        with torch.enable_grad():
            x2 = x.detach().requires_grad_(True)
            w2 = weight.detach().requires_grad_(True)
            b2 = bias.detach().requires_grad_(True) if bias is not None else None
            y = F.layer_norm(x2, (x.shape[-1],), w2, b2, ctx.eps)
            if b2 is not None:
                gx, gw, gb = torch.autograd.grad(y, (x2, w2, b2), grad_output)
            else:
                gx, gw = torch.autograd.grad(y, (x2, w2), grad_output)
                gb = None
        return gx, gw, gb, None


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    if not x.is_cuda:
        return _fallback(x, weight, bias, eps)
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)):
        return _LayerNormFunction.apply(x, weight, bias, eps)
    if not _ensure():
        return _fallback(x, weight, bias, eps)
    try:
        return torch.ops.cuda_qwen3_vl.layernorm_forward(
            x.contiguous(), weight.contiguous(),
            bias.contiguous() if bias is not None else None, eps
        )
    except Exception as exc:
        maybe_strict_raise("layernorm", exc)
        return _fallback(x, weight, bias, eps)
