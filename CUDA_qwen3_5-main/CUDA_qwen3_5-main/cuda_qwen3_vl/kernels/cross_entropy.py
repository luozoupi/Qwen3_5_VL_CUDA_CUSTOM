from __future__ import annotations

import torch
import torch.nn.functional as F

from ._loader import load_op, maybe_strict_raise


def _ensure() -> bool:
    return load_op("cross_entropy")


def _fallback(logits, targets, ignore_index):
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


class _CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index):
        if not _ensure():
            return _fallback(logits, targets, ignore_index)
        try:
            loss_per_token, lse = torch.ops.cuda_qwen3_vl.cross_entropy_forward(
                logits.contiguous(), targets.contiguous(), ignore_index
            )
            valid = (targets != ignore_index).sum().clamp(min=1)
            loss = loss_per_token.sum() / valid
            ctx.save_for_backward(logits, targets, lse)
            ctx.ignore_index = ignore_index
            ctx.n_valid = valid.item()
            return loss
        except Exception as exc:
            maybe_strict_raise("cross_entropy", exc)
            return _fallback(logits, targets, ignore_index)

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, lse = ctx.saved_tensors
        grad_scale = grad_output.item() / ctx.n_valid
        try:
            grad_logits = torch.ops.cuda_qwen3_vl.cross_entropy_backward(
                logits.contiguous(), targets.contiguous(), lse.contiguous(),
                grad_scale, ctx.ignore_index
            )
            return grad_logits, None, None
        except Exception as exc:
            maybe_strict_raise("cross_entropy", exc)
            with torch.enable_grad():
                l2 = logits.detach().requires_grad_(True)
                loss = F.cross_entropy(l2, targets, ignore_index=ctx.ignore_index)
                (g,) = torch.autograd.grad(loss, l2, grad_output)
            return g, None, None


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    if not logits.is_cuda:
        return _fallback(logits, targets, ignore_index)
    if torch.is_grad_enabled() and logits.requires_grad:
        return _CrossEntropyFunction.apply(logits, targets, ignore_index)
    if not _ensure():
        return _fallback(logits, targets, ignore_index)
    try:
        loss_per_token, _ = torch.ops.cuda_qwen3_vl.cross_entropy_forward(
            logits.contiguous(), targets.contiguous(), ignore_index
        )
        valid = (targets != ignore_index).sum().clamp(min=1)
        return loss_per_token.sum() / valid
    except Exception as exc:
        maybe_strict_raise("cross_entropy", exc)
        return _fallback(logits, targets, ignore_index)
