"""Multimodal RoPE: 3D position (T, H, W) interleaved across head dim.

The HF `apply_multimodal_rotary_pos_emb` expects cos/sin shaped (3, B, S, D_rope) and
a `mrope_section` list specifying how many elements each axis contributes. The CUDA
kernel here consumes *pre-interleaved* cos/sin of shape (B, S, D_rope), which is
obtained by selecting slices from each axis per section and concatenating.
"""
from __future__ import annotations

import torch

from ._loader import load_op, maybe_strict_raise


def _ensure() -> bool:
    return load_op("mrope")


def _interleave_cos_sin(cos_3d: torch.Tensor, sin_3d: torch.Tensor, mrope_section: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble (B, S, D_rope) cos/sin from (3, B, S, D_rope) using mrope_section.

    HF applies: cos = [cos_axis[section_i] for (section_i, axis) in zip(sections*2, [0,1,2,0,1,2,...])]
    Effectively: split last dim into 2 halves (for rotate_half), each half has `sections` slices from the 3 axes.
    """
    # cos_3d[axis, B, S, D_rope]
    mrope_section = mrope_section * 2  # repeated for the two halves
    cos_parts = []
    sin_parts = []
    offset = 0
    for i, m in enumerate(mrope_section):
        axis = i % 3
        cos_parts.append(cos_3d[axis, ..., offset:offset + m])
        sin_parts.append(sin_3d[axis, ..., offset:offset + m])
        offset += m
    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


def _fallback(x, cos, sin):
    # Same rotate_half as standard RoPE
    D_rope = cos.shape[-1]
    half = D_rope // 2
    x_rot = x[..., :D_rope]
    x_pass = x[..., D_rope:]
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    c = cos.unsqueeze(1)  # (B, 1, S, D_rope)
    s = sin.unsqueeze(1)
    c1, s1 = c[..., :half], s[..., :half]
    rot = torch.cat([x1 * c1 - x2 * s1, x2 * c1 + x1 * s1], dim=-1)
    return torch.cat([rot, x_pass], dim=-1)


class _MRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        if not _ensure():
            return _fallback(x, cos, sin)
        try:
            return torch.ops.cuda_qwen3_vl.mrope_forward(x.contiguous(), cos.contiguous(), sin.contiguous())
        except Exception as exc:
            maybe_strict_raise("mrope", exc)
            return _fallback(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        if not _ensure():
            return _fallback(grad_output, cos, -sin), None, None
        try:
            g = torch.ops.cuda_qwen3_vl.mrope_forward(
                grad_output.contiguous(), cos.contiguous(), (-sin).contiguous()
            )
            return g, None, None
        except Exception as exc:
            maybe_strict_raise("mrope", exc)
            return _fallback(grad_output, cos, -sin), None, None


def apply_mrope(
    x: torch.Tensor,
    cos_3d: torch.Tensor,
    sin_3d: torch.Tensor,
    mrope_section: list[int],
) -> torch.Tensor:
    """Apply MRoPE to (B, H, S, D) tensor.

    cos_3d / sin_3d: (3, B, S, D_rope). mrope_section sums to D_rope // 2.
    """
    cos, sin = _interleave_cos_sin(cos_3d, sin_3d, mrope_section)
    # Kernel requires cos/sin dtype to match x; cast before dispatch so the CUDA path is usable.
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    if not x.is_cuda:
        return _fallback(x, cos, sin)
    if torch.is_grad_enabled() and x.requires_grad:
        return _MRopeFunction.apply(x, cos, sin)
    if not _ensure():
        return _fallback(x, cos, sin)
    try:
        return torch.ops.cuda_qwen3_vl.mrope_forward(x.contiguous(), cos.contiguous(), sin.contiguous())
    except Exception as exc:
        maybe_strict_raise("mrope", exc)
        return _fallback(x, cos, sin)
