"""Sparse MoE block for Qwen3-VL-MoE.

Matches HF Qwen3VLMoeTextExperts / Qwen3VLMoeTextTopKRouter layout:
- gate: (num_experts, hidden_size) — row-major linear router
- experts.gate_up_proj: (num_experts, 2 * moe_intermediate_size, hidden_size)
- experts.down_proj: (num_experts, hidden_size, moe_intermediate_size)

Routing: softmax(x @ gate.T) -> topk -> sigmoid-gating via experts.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cuda_qwen3_vl.kernels import (
    cuda_batched_gemm,
    cuda_index_add,
    cuda_topk,
    matmul,
    softmax,
    swiglu,
)


class CudaSparseMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

        # Router
        self.gate_weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        # Experts (HF layout)
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, moe_intermediate_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.gate_weight, std=0.02)
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, H = x.shape
        x_flat = x.reshape(-1, H)  # (BS, H)
        N = x_flat.size(0)

        # Router
        router_logits = matmul(x_flat, self.gate_weight)  # (BS, E)
        routing_weights = softmax(router_logits.float())
        top_vals, top_idx = cuda_topk(routing_weights, self.top_k)
        if self.norm_topk_prob:
            top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
        top_vals = top_vals.to(x_flat.dtype)

        # Expert dispatch: for each expert, gather its tokens, matmul, scatter-add back.
        # cuda_index_add is autograd-aware: when grads are enabled it clones rather than
        # mutating, so we chain the running `final` tensor through it.
        final = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            mask = top_idx == e
            if not mask.any():
                continue
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            x_sub = x_flat[token_idx]
            gu = matmul(x_sub, self.gate_up_proj[e])
            gate_part = gu[:, :self.moe_intermediate_size]
            up_part = gu[:, self.moe_intermediate_size:]
            activated = swiglu(gate_part, up_part)
            expert_out = matmul(activated, self.down_proj[e])
            weighted = expert_out * top_vals[token_idx, slot_idx, None]
            final = cuda_index_add(final, weighted.to(x_flat.dtype), token_idx)

        return final.reshape(B, S, H), router_logits
