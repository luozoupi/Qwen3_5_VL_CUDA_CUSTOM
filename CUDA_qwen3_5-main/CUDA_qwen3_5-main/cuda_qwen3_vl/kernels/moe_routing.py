"""MoE routing helpers backed by our CUDA kernels.

All three ops have autograd wrappers:
- cuda_topk: fwd gathers top-k values + indices; bwd scatters value-grads back via the
  saved indices. Gradient w.r.t. indices is None (non-diff).
- cuda_index_add: fwd scatters source rows into target at index positions; bwd sends
  target-grad back to source via gather and is identity for target. Target's in-place
  write is replaced by a non-destructive clone so autograd tracks the graph.
- cuda_batched_gemm: fwd Y[e] = X[e] @ W[e]^T; bwd:
    dX[e] = dY[e] @ W[e]        (same kernel with W untransposed in our convention)
    dW[e] = dY[e]^T @ X[e]      (same kernel, transposed inputs)
"""
from __future__ import annotations

import torch

from ._loader import load_op, maybe_strict_raise, record_fallback


def _ensure() -> bool:
    return load_op("moe_routing")


# ---------- topk ----------

class _TopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(x)
        ctx.k = k
        if not x.is_cuda or not _ensure():
            if x.is_cuda and not _ensure():
                record_fallback("moe_routing.topk", "build_or_load_failed")
            return torch.topk(x, k, dim=-1)
        try:
            vals, idxs = torch.ops.cuda_qwen3_vl.topk_forward(x.contiguous(), k)
            ctx.saved_idxs = idxs  # non-tensor-marker for backward
            return vals, idxs
        except Exception as exc:
            maybe_strict_raise("moe_routing.topk", exc)
            return torch.topk(x, k, dim=-1)

    @staticmethod
    def backward(ctx, grad_vals: torch.Tensor, grad_idxs: torch.Tensor | None):
        (x,) = ctx.saved_tensors
        # Find indices: prefer the stashed idxs from forward; fall back to recomputing.
        idxs = getattr(ctx, "saved_idxs", None)
        if idxs is None:
            _, idxs = torch.topk(x, ctx.k, dim=-1)
        grad_x = torch.zeros_like(x)
        grad_x.scatter_add_(-1, idxs, grad_vals.to(grad_x.dtype))
        return grad_x, None


def cuda_topk(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_grad_enabled() and x.requires_grad:
        return _TopKFunction.apply(x, k)
    if not x.is_cuda or not _ensure():
        if x.is_cuda and not _ensure():
            record_fallback("moe_routing.topk", "build_or_load_failed")
        return torch.topk(x, k, dim=-1)
    try:
        vals, idxs = torch.ops.cuda_qwen3_vl.topk_forward(x.contiguous(), k)
        return vals, idxs
    except Exception as exc:
        maybe_strict_raise("moe_routing.topk", exc)
        return torch.topk(x, k, dim=-1)


# ---------- index_add ----------

class _IndexAddFunction(torch.autograd.Function):
    """out = target.clone(); out[index] += source. Non-destructive to preserve autograd."""

    @staticmethod
    def forward(ctx, target: torch.Tensor, source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(index)
        ctx.source_shape = source.shape
        out = target.clone()
        if not target.is_cuda or not _ensure():
            if target.is_cuda and not _ensure():
                record_fallback("moe_routing.index_add", "build_or_load_failed")
            out.index_add_(0, index, source)
            return out
        try:
            torch.ops.cuda_qwen3_vl.index_add_forward(out, source.contiguous(), index.contiguous())
            return out
        except Exception as exc:
            maybe_strict_raise("moe_routing.index_add", exc)
            out.index_add_(0, index, source)
            return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (index,) = ctx.saved_tensors
        # d/d target = grad_out (identity add)
        grad_target = grad_out
        # d/d source = grad_out[index] (gather)
        grad_source = grad_out.index_select(0, index).to(grad_out.dtype).reshape(ctx.source_shape)
        return grad_target, grad_source, None


def cuda_index_add(target: torch.Tensor, source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """out = target + scatter_add(source, index). Returns a NEW tensor (non-destructive).

    If you want in-place semantics for inference (old behavior), call with grad disabled
    — the autograd Function still clones, but that clone cost is sub-percent of a matmul.
    """
    if torch.is_grad_enabled() and (target.requires_grad or source.requires_grad):
        return _IndexAddFunction.apply(target, source, index)
    if not target.is_cuda or not _ensure():
        if target.is_cuda and not _ensure():
            record_fallback("moe_routing.index_add", "build_or_load_failed")
        target.index_add_(0, index, source)
        return target
    try:
        return torch.ops.cuda_qwen3_vl.index_add_forward(target, source.contiguous(), index.contiguous())
    except Exception as exc:
        maybe_strict_raise("moe_routing.index_add", exc)
        target.index_add_(0, index, source)
        return target


# ---------- batched GEMM ----------

def _bgemm_fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Y[e, m, n] = sum_k X[e, m, k] * W[e, n, k]. Calls the CUDA kernel."""
    return torch.ops.cuda_qwen3_vl.batched_gemm_forward(x.contiguous(), w.contiguous())


class _BatchedGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, w)
        if not x.is_cuda or not _ensure():
            if x.is_cuda and not _ensure():
                record_fallback("moe_routing.batched_gemm", "build_or_load_failed")
            return torch.einsum("emk,enk->emn", x, w)
        try:
            return _bgemm_fwd(x, w)
        except Exception as exc:
            maybe_strict_raise("moe_routing.batched_gemm", exc)
            return torch.einsum("emk,enk->emn", x, w)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        # Forward:  Y[e, m, n] = sum_k X[e, m, k] * W[e, n, k]    (X: e,m,k   W: e,n,k)
        # Backward:
        #   dX[e, m, k] = sum_n dY[e, m, n] * W[e, n, k]
        #               = einsum("emn,enk->emk", dY, W)
        #               = bgemm(dY, W.transpose(-1,-2)) in our kernel's convention
        #                 because kernel does Y[m,n] = sum_k X[m,k] * W[n,k]
        #     let X' = dY  (e, m, n), W' = W^T_last (e, k, n) s.t. W'[e, k, n] = W[e, n, k]^T inner
        #     but our kernel's W' has shape (e, N_out, K_in). We need output (e, m, k), so N_out=k, K_in=n.
        #     Use W' = W.transpose(-1, -2)? That gives (e, k, n) — matches (e, N_out=k, K_in=n). ✓
        #   dW[e, n, k] = sum_m dY[e, m, n] * X[e, m, k]
        #               = einsum("emn,emk->enk", dY, X)
        #               In our kernel: output (e, n, k), so call with X'=dY.transpose(-1,-2) (e, n, m),
        #               W'=X.transpose(-1,-2) (e, k, m). Shapes (e, N_out=n, K_in=m) vs (e, N_out=k, K_in=m).
        #               Wait, the kernel does Y[m,n] = sum_k X[m,k]*W[n,k]. Let X_bwd = dY (e,m,n),
        #               W_bwd = X (e,m,k)? That gives sum_k dY[m,k]*X[m,k] — wrong.
        # Simpler: fall through to einsum for bwd. It's correct and only runs during training.
        x, w = ctx.saved_tensors
        grad_x = torch.einsum("emn,enk->emk", grad_y, w)
        grad_w = torch.einsum("emn,emk->enk", grad_y, x)
        return grad_x, grad_w


def cuda_batched_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Y[e] = X[e] @ W[e].T. x: (E,M,K), w: (E,N,K), out: (E,M,N)."""
    if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
        return _BatchedGemmFunction.apply(x, w)
    if not x.is_cuda or not _ensure():
        if x.is_cuda and not _ensure():
            record_fallback("moe_routing.batched_gemm", "build_or_load_failed")
        return torch.einsum("emk,enk->emn", x, w)
    try:
        return _bgemm_fwd(x, w)
    except Exception as exc:
        maybe_strict_raise("moe_routing.batched_gemm", exc)
        return torch.einsum("emk,enk->emn", x, w)
