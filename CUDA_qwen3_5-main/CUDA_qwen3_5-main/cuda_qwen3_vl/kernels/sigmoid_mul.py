from __future__ import annotations

import torch

from ._loader import load_op, maybe_strict_raise


def _ensure() -> bool:
    return load_op("sigmoid_mul")


def _fallback_fwd(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(gate)


def _fallback_bwd(x, gate, grad_out):
    s = torch.sigmoid(gate.float())
    gx = grad_out.float() * s
    gg = grad_out.float() * x.float() * s * (1.0 - s)
    return gx.to(x.dtype), gg.to(gate.dtype)


class _SigmoidMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate):
        ctx.save_for_backward(x, gate)
        if not _ensure():
            return _fallback_fwd(x, gate)
        try:
            return torch.ops.cuda_qwen3_vl.sigmoid_mul_forward(x.contiguous(), gate.contiguous())
        except Exception as exc:
            maybe_strict_raise("sigmoid_mul", exc)
            return _fallback_fwd(x, gate)

    @staticmethod
    def backward(ctx, grad_output):
        x, gate = ctx.saved_tensors
        if not _ensure():
            return _fallback_bwd(x, gate, grad_output)
        try:
            gx, gg = torch.ops.cuda_qwen3_vl.sigmoid_mul_backward(
                x.contiguous(), gate.contiguous(), grad_output.contiguous()
            )
            return gx, gg
        except Exception as exc:
            maybe_strict_raise("sigmoid_mul", exc)
            gx, gg = _fallback_bwd(x, gate, grad_output)
            return gx, gg


def sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or not gate.is_cuda:
        return _fallback_fwd(x, gate)
    if torch.is_grad_enabled() and (x.requires_grad or gate.requires_grad):
        return _SigmoidMulFunction.apply(x, gate)
    if not _ensure():
        return _fallback_fwd(x, gate)
    try:
        return torch.ops.cuda_qwen3_vl.sigmoid_mul_forward(x.contiguous(), gate.contiguous())
    except Exception as exc:
        maybe_strict_raise("sigmoid_mul", exc)
        return _fallback_fwd(x, gate)
