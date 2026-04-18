"""Text decoder layer for Qwen3-VL (dense and MoE variants)."""
from __future__ import annotations

import torch
from torch import nn

from cuda_qwen3_vl.kernels import residual_add
from cuda_qwen3_vl.modules.attention import CudaFullAttention
from cuda_qwen3_vl.modules.mlp import CudaSwiGLUMLP
from cuda_qwen3_vl.modules.moe import CudaSparseMoE
from cuda_qwen3_vl.modules.norms import CudaRMSNorm


class CudaTextDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        use_moe: bool = False,
        num_experts: int = 0,
        top_k: int = 0,
        moe_intermediate_size: int = 0,
        attention_bias: bool = False,
    ) -> None:
        super().__init__()
        self.input_layernorm = CudaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = CudaFullAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
        )
        self.post_attention_layernorm = CudaRMSNorm(hidden_size, eps=rms_norm_eps)
        if use_moe:
            self.mlp = CudaSparseMoE(
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
            )
        else:
            self.mlp = CudaSwiGLUMLP(hidden_size, intermediate_size)
        self.use_moe = use_moe

    def forward(
        self,
        x: torch.Tensor,
        mrope_apply,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.self_attn(self.input_layernorm(x), mrope_apply, past_kv=past_kv)
        x = residual_add(attn_out, x)
        mlp_in = self.post_attention_layernorm(x)
        if self.use_moe:
            mlp_out, router_logits = self.mlp(mlp_in)
        else:
            mlp_out, router_logits = self.mlp(mlp_in), None
        x = residual_add(mlp_out, x)
        return x, router_logits, new_kv
