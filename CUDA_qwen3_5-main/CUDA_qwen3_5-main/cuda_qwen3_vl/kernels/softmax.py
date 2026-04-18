from __future__ import annotations

import torch

from ._loader import load_op, maybe_strict_raise


def _ensure() -> bool:
    return load_op("softmax")


class _SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if not _ensure():
            out = torch.softmax(x, dim=-1)
        else:
            try:
                out = torch.ops.cuda_qwen3_vl.softmax_forward(x.contiguous())
            except Exception as exc:
                maybe_strict_raise("softmax", exc)
                out = torch.softmax(x, dim=-1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        if not _ensure():
            dot = (out * grad_output).sum(dim=-1, keepdim=True)
            return out * (grad_output - dot)
        try:
            return torch.ops.cuda_qwen3_vl.softmax_backward(out.contiguous(), grad_output.contiguous())
        except Exception as exc:
            maybe_strict_raise("softmax", exc)
            dot = (out * grad_output).sum(dim=-1, keepdim=True)
            return out * (grad_output - dot)


def softmax(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        return torch.softmax(x, dim=-1)
    if torch.is_grad_enabled() and x.requires_grad:
        return _SoftmaxFunction.apply(x)
    if not _ensure():
        return torch.softmax(x, dim=-1)
    try:
        return torch.ops.cuda_qwen3_vl.softmax_forward(x.contiguous())
    except Exception as exc:
        maybe_strict_raise("softmax", exc)
        return torch.softmax(x, dim=-1)
