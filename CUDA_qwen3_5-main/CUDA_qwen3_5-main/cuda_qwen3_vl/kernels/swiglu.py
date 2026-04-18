from __future__ import annotations

import torch
import torch.nn.functional as F

from ._loader import load_op, maybe_strict_raise


def _ensure() -> bool:
    return load_op("swiglu")


def _fallback_fwd(gate, up):
    return F.silu(gate) * up


def _fallback_bwd(gate, up, grad_out):
    g = gate.float()
    u = up.float()
    go = grad_out.float()
    sig = torch.sigmoid(g)
    silu = g * sig
    dsilu_dg = sig * (1.0 + g * (1.0 - sig))
    return (go * u * dsilu_dg).to(gate.dtype), (go * silu).to(up.dtype)


class _SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        if not _ensure():
            return _fallback_fwd(gate, up)
        try:
            return torch.ops.cuda_qwen3_vl.swiglu_forward(gate.contiguous(), up.contiguous())
        except Exception as exc:
            maybe_strict_raise("swiglu", exc)
            return _fallback_fwd(gate, up)

    @staticmethod
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        if not _ensure():
            gg, gu = _fallback_bwd(gate, up, grad_output)
            return gg, gu
        try:
            gg, gu = torch.ops.cuda_qwen3_vl.swiglu_backward(
                gate.contiguous(), up.contiguous(), grad_output.contiguous()
            )
            return gg, gu
        except Exception as exc:
            maybe_strict_raise("swiglu", exc)
            gg, gu = _fallback_bwd(gate, up, grad_output)
            return gg, gu


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if not gate.is_cuda or not up.is_cuda:
        return _fallback_fwd(gate, up)
    if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad):
        return _SwiGLUFunction.apply(gate, up)
    if not _ensure():
        return _fallback_fwd(gate, up)
    try:
        return torch.ops.cuda_qwen3_vl.swiglu_forward(gate.contiguous(), up.contiguous())
    except Exception as exc:
        maybe_strict_raise("swiglu", exc)
        return _fallback_fwd(gate, up)
