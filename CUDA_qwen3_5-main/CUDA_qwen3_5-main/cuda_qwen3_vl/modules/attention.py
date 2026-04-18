from __future__ import annotations

import torch
from torch import nn

from cuda_qwen3_vl.kernels import apply_rope, flash_attention
from cuda_qwen3_vl.modules.linear import CudaLinear
from cuda_qwen3_vl.modules.norms import CudaRMSNorm


class CudaVisionAttention(nn.Module):
    """Qwen3-VL vision attention: non-causal, Q/K/V/O projections with bias.

    HF stores qkv as a single fused linear (3*hidden) — we match that layout.
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = CudaLinear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = CudaLinear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # (B, S, H, D) -> (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = flash_attention(q, k, v, scale=self.scale, is_causal=False, num_kv_groups=1)
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.proj(out)


class CudaFullAttention(nn.Module):
    """Qwen3-VL text attention: causal GQA with per-head RMSNorm + MRoPE.

    Supports KV cache: pass `past_kv=(past_k, past_v)` of shape (B, H_kv, S_cache, D)
    to run an incremental decode step, and get back the updated cache.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = self.head_dim ** -0.5

        self.q_proj = CudaLinear(hidden_size, num_heads * self.head_dim, bias=attention_bias)
        self.k_proj = CudaLinear(hidden_size, num_kv_heads * self.head_dim, bias=attention_bias)
        self.v_proj = CudaLinear(hidden_size, num_kv_heads * self.head_dim, bias=attention_bias)
        self.o_proj = CudaLinear(num_heads * self.head_dim, hidden_size, bias=attention_bias)
        self.q_norm = CudaRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = CudaRMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mrope_apply,  # callable (q, k) -> (q_rot, k_rot)
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = mrope_apply(q, k)
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v)
        # Causal only when Q and K have the same length (i.e. prefill with no cache).
        is_causal = q.shape[2] == k.shape[2]
        out = flash_attention(q, k, v, scale=self.scale, is_causal=is_causal, num_kv_groups=self.num_kv_groups)
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out), new_kv
