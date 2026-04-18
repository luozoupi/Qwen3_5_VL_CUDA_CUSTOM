"""HuggingFace weight loader for CUDA Qwen3-VL models.

Verified against the Qwen3-VL-8B-Instruct safetensors index. Exact HF key patterns:
- Text: `model.language_model.{embed_tokens,layers.N.*,norm}.weight`
- LM head: `lm_head.weight` (top-level, not under model.)
- Vision: `model.visual.patch_embed.proj.{weight,bias}`,
          `model.visual.pos_embed.weight`,
          `model.visual.blocks.N.{norm1,norm2,attn.{qkv,proj},mlp.{linear_fc1,linear_fc2}}.{weight,bias}`,
          `model.visual.merger.{norm,linear_fc1,linear_fc2}.{weight,bias}`,
          `model.visual.deepstack_merger_list.N.{norm,linear_fc1,linear_fc2}.{weight,bias}`
- MoE (Qwen3-VL-MoE variant): `model.language_model.layers.N.mlp.{experts.gate_up_proj,experts.down_proj,gate.weight}`
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open


_EXPERT_TRANSPOSE_KEYS = {
    # HF Qwen3-VL-MoE stores these transposed on dim (1, 2); see HF's
    # conversion_mapping.py `qwen3_vl_moe` rule. On-disk shapes:
    #   gate_up_proj: (E, H, 2*I)  -> target (E, 2*I, H)
    #   down_proj:    (E, I, H)    -> target (E, H, I)
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
}


def _needs_expert_transpose(hf_key: str) -> bool:
    return any(key in hf_key for key in _EXPERT_TRANSPOSE_KEYS)


def _map_name(name: str) -> str | None:
    """Translate HF parameter name to our internal name. Returns None if unmapped."""
    # LM head sits at top-level (not under model.)
    if name == "lm_head.weight":
        return "lm_head.weight"

    # Vision tower: prefix is `model.visual.`
    if name.startswith("model.visual."):
        rest = name[len("model.visual."):]  # e.g. "patch_embed.proj.weight"
        # patch_embed.proj.{w,b} -> patch_embed.{w,b}
        rest = rest.replace("patch_embed.proj.", "patch_embed.")
        # pos_embed.weight -> pos_embed.emb.weight (we wrapped nn.Embedding in a module)
        if rest == "pos_embed.weight":
            rest = "pos_embed.emb.weight"
        # deepstack_merger_list.N.* -> deepstack_mergers.N.*
        rest = rest.replace("deepstack_merger_list.", "deepstack_mergers.")
        # blocks.N.*, merger.* pass through unchanged
        return f"visual.{rest}"

    # Text stack: prefix is `model.language_model.`
    if name.startswith("model.language_model."):
        rest = name[len("model.language_model."):]
        # MoE-specific renames (only present in the MoE checkpoint):
        # layers.N.mlp.experts.gate_up_proj -> layers.N.mlp.gate_up_proj
        rest = rest.replace("mlp.experts.gate_up_proj", "mlp.gate_up_proj")
        rest = rest.replace("mlp.experts.down_proj", "mlp.down_proj")
        # layers.N.mlp.gate.weight -> layers.N.mlp.gate_weight
        rest = rest.replace("mlp.gate.weight", "mlp.gate_weight")
        return rest

    # Anything else: unmapped
    return None


def load_hf_weights(model: torch.nn.Module, snapshot_path: str | Path) -> dict:
    """Load safetensors from snapshot_path into model.

    Returns a dict with keys: 'loaded', 'missing', 'unexpected', 'mismatched'.
    """
    snapshot = Path(snapshot_path)
    files = sorted(snapshot.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors in {snapshot_path}")

    state_dict = dict(model.state_dict())
    remaining = set(state_dict.keys())
    loaded: list[str] = []
    unexpected: list[str] = []
    mismatched: list[tuple[str, tuple, tuple]] = []

    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                mapped = _map_name(key)
                if mapped is None or mapped not in state_dict:
                    unexpected.append(key)
                    continue
                tensor = sf.get_tensor(key)
                if _needs_expert_transpose(key):
                    tensor = tensor.transpose(1, 2).contiguous()
                target = state_dict[mapped]
                if tensor.shape != target.shape:
                    mismatched.append((key, tuple(tensor.shape), tuple(target.shape)))
                    continue
                with torch.no_grad():
                    target.copy_(tensor.to(target.dtype).to(target.device))
                loaded.append(mapped)
                remaining.discard(mapped)

    return {
        "loaded": loaded,
        "missing": sorted(remaining),
        "unexpected": unexpected,
        "mismatched": mismatched,
    }
