"""Config normalization for Qwen3-VL (dense + MoE)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VisionConfig:
    hidden_size: int = 1152
    num_layers: int = 27
    num_heads: int = 16
    intermediate_size: int = 4304
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    in_channels: int = 3
    num_position_embeddings: int = 2304
    out_hidden_size: int = 3584  # Language model hidden size (projected-to)
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    deepstack_layers: tuple[int, ...] = (8, 16, 24)


@dataclass
class TextConfig:
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    max_position_embeddings: int = 128000
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-6
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])
    attention_bias: bool = False
    tie_word_embeddings: bool = False


@dataclass
class MoETextConfig(TextConfig):
    num_experts: int = 60
    num_experts_per_tok: int = 4
    moe_intermediate_size: int = 1408
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] = field(default_factory=list)


@dataclass
class Qwen3VLConfig:
    family: str  # "dense" or "moe"
    vision: VisionConfig
    text: TextConfig | MoETextConfig

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "Qwen3VLConfig":
        """Normalize a HuggingFace Qwen3VLConfig / Qwen3VLMoeConfig into our layout."""
        raw = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
        vision_raw = raw.get("vision_config", {})
        text_raw = raw.get("text_config", raw)
        is_moe = (
            raw.get("model_type") == "qwen3_vl_moe"
            or int(text_raw.get("num_local_experts") or text_raw.get("num_experts") or 0) > 0
        )

        vision = VisionConfig(
            hidden_size=int(vision_raw.get("hidden_size", 1152)),
            num_layers=int(vision_raw.get("num_hidden_layers") or vision_raw.get("depth") or 27),
            num_heads=int(vision_raw.get("num_heads", 16)),
            intermediate_size=int(vision_raw.get("intermediate_size", 4304)),
            patch_size=int(vision_raw.get("patch_size", 16)),
            temporal_patch_size=int(vision_raw.get("temporal_patch_size", 2)),
            spatial_merge_size=int(vision_raw.get("spatial_merge_size", 2)),
            in_channels=int(vision_raw.get("in_channels", 3)),
            num_position_embeddings=int(vision_raw.get("num_position_embeddings", 2304)),
            out_hidden_size=int(vision_raw.get("out_hidden_size") or text_raw.get("hidden_size", 3584)),
            rope_theta=float(vision_raw.get("rope_theta", 10000.0)),
            rms_norm_eps=float(vision_raw.get("rms_norm_eps", 1e-6)),
            deepstack_layers=tuple(vision_raw.get("deepstack_visual_indexes", (8, 16, 24))),
        )

        common_text = dict(
            hidden_size=int(text_raw.get("hidden_size", 4096)),
            intermediate_size=int(text_raw.get("intermediate_size", 22016)),
            num_layers=int(text_raw.get("num_hidden_layers", 32)),
            num_heads=int(text_raw.get("num_attention_heads", 32)),
            num_kv_heads=int(text_raw.get("num_key_value_heads", 8)),
            head_dim=int(text_raw.get("head_dim") or
                         (int(text_raw.get("hidden_size", 4096)) // int(text_raw.get("num_attention_heads", 32)))),
            vocab_size=int(text_raw.get("vocab_size", 151936)),
            max_position_embeddings=int(text_raw.get("max_position_embeddings", 128000)),
            rope_theta=float(text_raw.get("rope_theta", 500000.0)),
            rms_norm_eps=float(text_raw.get("rms_norm_eps", 1e-6)),
            mrope_section=list(text_raw.get("rope_scaling", {}).get("mrope_section", [24, 20, 20])),
            attention_bias=bool(text_raw.get("attention_bias", False)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        )

        if is_moe:
            text = MoETextConfig(
                **common_text,
                num_experts=int(text_raw.get("num_local_experts") or text_raw.get("num_experts") or 60),
                num_experts_per_tok=int(text_raw.get("num_experts_per_tok", 4)),
                moe_intermediate_size=int(text_raw.get("moe_intermediate_size", 1408)),
                norm_topk_prob=bool(text_raw.get("norm_topk_prob", True)),
                decoder_sparse_step=int(text_raw.get("decoder_sparse_step", 1)),
                mlp_only_layers=list(text_raw.get("mlp_only_layers", [])),
            )
        else:
            text = TextConfig(**common_text)

        return cls(family="moe" if is_moe else "dense", vision=vision, text=text)
