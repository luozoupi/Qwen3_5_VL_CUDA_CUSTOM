from __future__ import annotations

import torch

from ._loader import load_op, maybe_strict_raise


def _ensure() -> bool:
    return load_op("rope")


def _fallback(x, cos, sin):
    # x: (B, H, S, D). cos/sin: (B, S, D_rope) or (S, D_rope)
    D_rope = cos.shape[-1]
    half = D_rope // 2
    x_rot = x[..., :D_rope]
    x_pass = x[..., D_rope:]
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    if cos.dim() == 3:
        c = cos.unsqueeze(1)  # (B, 1, S, D_rope)
        s = sin.unsqueeze(1)
    else:
        c = cos.unsqueeze(0).unsqueeze(0)
        s = sin.unsqueeze(0).unsqueeze(0)
    c1, s1 = c[..., :half], s[..., :half]
    rot = torch.cat([x1 * c1 - x2 * s1, x2 * c1 + x1 * s1], dim=-1)
    return torch.cat([rot, x_pass], dim=-1)


class _RopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        if not _ensure():
            return _fallback(x, cos, sin)
        try:
            return torch.ops.cuda_qwen3_vl.rope_forward(x.contiguous(), cos.contiguous(), sin.contiguous())
        except Exception as exc:
            maybe_strict_raise("rope", exc)
            return _fallback(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        # RoPE backward is forward with negated sin
        if not _ensure():
            return _fallback(grad_output, cos, -sin), None, None
        try:
            return torch.ops.cuda_qwen3_vl.rope_forward(
                grad_output.contiguous(), cos.contiguous(), (-sin).contiguous()
            ), None, None
        except Exception as exc:
            maybe_strict_raise("rope", exc)
            return _fallback(grad_output, cos, -sin), None, None


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Kernel requires cos/sin dtype to match x; cast before dispatch so the CUDA path is usable.
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    if not x.is_cuda:
        return _fallback(x, cos, sin)
    if torch.is_grad_enabled() and x.requires_grad:
        return _RopeFunction.apply(x, cos, sin)
    if not _ensure():
        return _fallback(x, cos, sin)
    try:
        return torch.ops.cuda_qwen3_vl.rope_forward(x.contiguous(), cos.contiguous(), sin.contiguous())
    except Exception as exc:
        maybe_strict_raise("rope", exc)
        return _fallback(x, cos, sin)
