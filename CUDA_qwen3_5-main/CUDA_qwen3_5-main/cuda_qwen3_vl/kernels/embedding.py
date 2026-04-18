from __future__ import annotations

import torch
import torch.nn.functional as F

from ._loader import load_op, maybe_strict_raise


def _ensure() -> bool:
    return load_op("embedding")


class _EmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, weight, padding_idx):
        ctx.save_for_backward(ids)
        ctx.num_embeddings = weight.size(0)
        ctx.padding_idx = padding_idx if padding_idx is not None else -1
        if not _ensure():
            return F.embedding(ids, weight, padding_idx=padding_idx)
        try:
            return torch.ops.cuda_qwen3_vl.embedding_forward(ids.contiguous(), weight.contiguous())
        except Exception as exc:
            maybe_strict_raise("embedding", exc)
            return F.embedding(ids, weight, padding_idx=padding_idx)

    @staticmethod
    def backward(ctx, grad_output):
        (ids,) = ctx.saved_tensors
        if not _ensure():
            # Fallback via autograd
            w = torch.zeros(ctx.num_embeddings, grad_output.size(-1),
                            device=grad_output.device, dtype=grad_output.dtype,
                            requires_grad=True)
            with torch.enable_grad():
                y = F.embedding(ids, w, padding_idx=None if ctx.padding_idx < 0 else ctx.padding_idx)
                (gw,) = torch.autograd.grad(y, w, grad_output)
            return None, gw, None
        try:
            gw = torch.ops.cuda_qwen3_vl.embedding_backward(
                ids.contiguous(), grad_output.contiguous(),
                ctx.num_embeddings, ctx.padding_idx
            )
            return None, gw, None
        except Exception as exc:
            maybe_strict_raise("embedding", exc)
            w = torch.zeros(ctx.num_embeddings, grad_output.size(-1),
                            device=grad_output.device, dtype=grad_output.dtype,
                            requires_grad=True)
            with torch.enable_grad():
                y = F.embedding(ids, w, padding_idx=None if ctx.padding_idx < 0 else ctx.padding_idx)
                (gw,) = torch.autograd.grad(y, w, grad_output)
            return None, gw, None


def embedding(ids: torch.Tensor, weight: torch.Tensor, padding_idx: int | None = None) -> torch.Tensor:
    if not weight.is_cuda:
        return F.embedding(ids, weight, padding_idx=padding_idx)
    if torch.is_grad_enabled() and weight.requires_grad:
        return _EmbeddingFunction.apply(ids, weight, padding_idx)
    if not _ensure():
        return F.embedding(ids, weight, padding_idx=padding_idx)
    try:
        return torch.ops.cuda_qwen3_vl.embedding_forward(ids.contiguous(), weight.contiguous())
    except Exception as exc:
        maybe_strict_raise("embedding", exc)
        return F.embedding(ids, weight, padding_idx=padding_idx)
