"""Qwen3-VL dense conditional-generation model (e.g. 8B)."""
from __future__ import annotations

import torch
from torch import nn

from cuda_qwen3_vl.configs import Qwen3VLConfig
from cuda_qwen3_vl.modules import (
    CudaEmbedding,
    CudaLinear,
    CudaRMSNorm,
    CudaTextDecoderLayer,
    TextMRoPE,
)
from cuda_qwen3_vl.models.common import CudaVisionTower


class CudaQwen3VLDenseModel(nn.Module):
    """Dense Qwen3-VL with full CUDA text stack. Supports KV-cache for generation."""

    def __init__(self, cfg: Qwen3VLConfig) -> None:
        super().__init__()
        assert cfg.family == "dense", f"expected dense config, got {cfg.family}"
        self.cfg = cfg
        self.visual = CudaVisionTower(cfg.vision)
        t = cfg.text
        self.embed_tokens = CudaEmbedding(t.vocab_size, t.hidden_size)
        self.rotary = TextMRoPE(t.head_dim, theta=t.rope_theta, mrope_section=t.mrope_section)
        self.layers = nn.ModuleList([
            CudaTextDecoderLayer(
                hidden_size=t.hidden_size,
                num_heads=t.num_heads,
                num_kv_heads=t.num_kv_heads,
                head_dim=t.head_dim,
                intermediate_size=t.intermediate_size,
                rms_norm_eps=t.rms_norm_eps,
                use_moe=False,
                attention_bias=t.attention_bias,
            )
            for _ in range(t.num_layers)
        ])
        self.norm = CudaRMSNorm(t.hidden_size, eps=t.rms_norm_eps)
        self.lm_head = CudaLinear(t.hidden_size, t.vocab_size, bias=False)
        if t.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks, :] = hidden_states[visual_pos_masks, :] + visual_embeds
        return hidden_states

    def _text_forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids_3d: torch.Tensor,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = inputs_embeds
        mrope_fn = lambda q, k: self.rotary.apply(q, k, position_ids_3d)
        n_deepstack = len(deepstack_visual_embeds) if deepstack_visual_embeds is not None else 0
        new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, _, new_kv = layer(x, mrope_fn, past_kv=layer_past)
            new_kvs.append(new_kv)
            if deepstack_visual_embeds is not None and visual_pos_masks is not None and i < n_deepstack:
                x = self._deepstack_process(x, visual_pos_masks, deepstack_visual_embeds[i])
        return self.norm(x), new_kvs

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Same signature as before plus `past_key_values` + `return_kv`.

        When `past_key_values` is provided, inputs should be ONLY the new tokens
        (typically a single token during decode), and `position_ids` should be the
        positions of those new tokens.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids / inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            B, S = inputs_embeds.shape[:2]
            past_len = past_key_values[0][0].shape[2] if past_key_values else 0
            pos = torch.arange(past_len, past_len + S, device=inputs_embeds.device).unsqueeze(0).expand(B, S)
            position_ids_3d = torch.stack([pos, pos, pos], dim=0)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids_3d = position_ids[1:]
        elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
            position_ids_3d = position_ids
        else:
            raise ValueError(f"Unsupported position_ids shape {tuple(position_ids.shape)}")

        hidden, new_kvs = self._text_forward(
            inputs_embeds, position_ids_3d,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(hidden)
        if return_kv:
            return logits, new_kvs
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        eos_token_id: int | None = None,
        position_ids: torch.Tensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Greedy decoding with KV-cache. Returns the full sequence (prompt + generated)."""
        # Prefill — populate caches
        logits, kv = self.forward(
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            return_kv=True,
        )
        next_tok = logits[:, -1].argmax(-1, keepdim=True)  # (B, 1)
        generated = torch.cat([input_ids, next_tok], dim=1)

        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break
            past_len = kv[0][0].shape[2]
            pos_1d = torch.tensor([[past_len]], device=input_ids.device, dtype=torch.long)
            pos_3d = torch.stack([pos_1d, pos_1d, pos_1d], dim=0)  # (3, 1, 1)
            logits, kv = self.forward(
                input_ids=next_tok,
                position_ids=pos_3d,
                past_key_values=kv,
                return_kv=True,
            )
            next_tok = logits[:, -1].argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated
