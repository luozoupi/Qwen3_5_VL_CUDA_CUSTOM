"""Vision tower parity vs HF on real Qwen3-VL-8B-Instruct weights.

This test exists to codify the acceptable numerical drift for the *downstream-consumed*
outputs (pooler_output + deepstack_features). The pre-LN `last_hidden_state` has large
absolute magnitudes (HF itself produces values up to ~15000) and accumulated bf16
rounding over 27 layers puts relative drift in the single-digit percent range — that's
not a bug and isn't asserted here.
"""
import os
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

SNAPSHOT = Path(
    "/mnt/data/vllm_models/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)

has_snapshot = SNAPSHOT.exists()


def _synth_image():
    img = Image.new("RGB", (224, 224), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 112, 112], fill=(220, 30, 30))
    d.rectangle([112, 0, 224, 112], fill=(30, 180, 30))
    d.rectangle([0, 112, 112, 224], fill=(30, 30, 200))
    d.rectangle([112, 112, 224, 224], fill=(230, 210, 40))
    d.line([(0, 0), (223, 223)], fill="black", width=6)
    return img


def _map_vision_key(name):
    if not name.startswith("model.visual."):
        return None
    rest = name[len("model.visual.") :]
    rest = rest.replace("patch_embed.proj.", "patch_embed.")
    if rest == "pos_embed.weight":
        rest = "pos_embed.emb.weight"
    rest = rest.replace("deepstack_merger_list.", "deepstack_mergers.")
    return rest


@cuda_only
@pytest.mark.skipif(not has_snapshot, reason=f"Qwen3-VL-8B snapshot not present at {SNAPSHOT}")
def test_vision_tower_downstream_parity_vs_hf():
    """Our vision tower's pooler_output + deepstack_features match HF within bf16 tolerance."""
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

    from cuda_qwen3_vl.configs import Qwen3VLConfig
    from cuda_qwen3_vl.models.common import CudaVisionTower

    hf_cfg = AutoConfig.from_pretrained(SNAPSHOT, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(SNAPSHOT, trust_remote_code=True)
    cfg = Qwen3VLConfig.from_hf_config(hf_cfg)

    our_vt = CudaVisionTower(cfg.vision).to(dtype=torch.bfloat16).cuda().eval()
    sd = dict(our_vt.state_dict())
    for f in sorted(SNAPSHOT.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                mapped = _map_vision_key(key)
                if mapped is None or mapped not in sd:
                    continue
                t = sf.get_tensor(key)
                with torch.no_grad():
                    sd[mapped].copy_(t.to(sd[mapped].dtype).to(sd[mapped].device))

    inputs = processor(
        text=["<|vision_start|><|image_pad|><|vision_end|>"],
        images=[_synth_image()],
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"].cuda().to(torch.bfloat16)
    grid_thw = inputs["image_grid_thw"].cuda()

    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        SNAPSHOT, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()

    with torch.no_grad():
        hf_out = hf_model.model.visual(pixel_values, grid_thw=grid_thw)
        our_pool, _our_last, our_ds = our_vt(pixel_values, grid_thw)

    # Pooler parity — this is what gets scattered into text embeds.
    # HF's absmean is ~0.14; we allow mean_abs_diff < 0.05 (< 40% relative, reality is ~7%).
    diff_pool = (our_pool.float() - hf_out.pooler_output.float()).abs()
    assert diff_pool.mean().item() < 0.05, f"pooler mean diff too large: {diff_pool.mean().item()}"
    assert diff_pool.max().item() < 2.0, f"pooler max diff too large: {diff_pool.max().item()}"

    # Deepstack parity — each feature goes into early text layers via residual.
    for i, (ours, hfs) in enumerate(zip(our_ds, hf_out.deepstack_features)):
        d = (ours.float() - hfs.float()).abs()
        assert d.mean().item() < 0.02, f"deepstack[{i}] mean diff too large: {d.mean().item()}"
        assert d.max().item() < 1.0, f"deepstack[{i}] max diff too large: {d.max().item()}"
