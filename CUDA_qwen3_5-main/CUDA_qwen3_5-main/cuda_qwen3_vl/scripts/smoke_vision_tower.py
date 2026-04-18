"""Vision tower parity: our CudaVisionTower vs HF Qwen3VLVisionModel.

Loads HF Qwen3-VL-8B-Instruct weights into both and compares:
- pooler_output (main merged features scattered into text)
- deepstack_features (list of 3 tensors for layers 8/16/24)
- last_hidden_state (pre-merger)
"""
from __future__ import annotations

import argparse
import sys

import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw

from cuda_qwen3_vl.configs import Qwen3VLConfig
from cuda_qwen3_vl.kernels import summarize_fallbacks
from cuda_qwen3_vl.models.common import CudaVisionTower


def _synthetic_image() -> Image.Image:
    img = Image.new("RGB", (224, 224), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 112, 112], fill=(220, 30, 30))
    d.rectangle([112, 0, 224, 112], fill=(30, 180, 30))
    d.rectangle([0, 112, 112, 224], fill=(30, 30, 200))
    d.rectangle([112, 112, 224, 224], fill=(230, 210, 40))
    d.line([(0, 0), (223, 223)], fill="black", width=6)
    return img


def _map_vision_key(name: str) -> str | None:
    """HF model prefixes vision keys with 'model.visual.'. Strip and remap the few
    renamed attrs. Returns None if key isn't a vision key."""
    if not name.startswith("model.visual."):
        return None
    rest = name[len("model.visual."):]
    rest = rest.replace("patch_embed.proj.", "patch_embed.")
    if rest == "pos_embed.weight":
        rest = "pos_embed.emb.weight"
    rest = rest.replace("deepstack_merger_list.", "deepstack_mergers.")
    return rest


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision tower parity")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"[1/5] Snapshot {args.model_id} ...")
    snapshot_path = snapshot_download(
        repo_id=args.model_id,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
    )

    print(f"[2/5] Load HF config + processor ...")
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    hf_cfg = AutoConfig.from_pretrained(snapshot_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(snapshot_path, trust_remote_code=True)
    cfg = Qwen3VLConfig.from_hf_config(hf_cfg)
    print(f"      vision hidden={cfg.vision.hidden_size} layers={cfg.vision.num_layers} heads={cfg.vision.num_heads} "
          f"deepstack={cfg.vision.deepstack_layers}")

    print(f"[3/5] Build our CudaVisionTower + load HF vision weights ...")
    torch.cuda.empty_cache()
    our_vt = CudaVisionTower(cfg.vision).to(dtype=dtype).cuda().eval()
    from safetensors import safe_open
    from pathlib import Path
    sd = dict(our_vt.state_dict())
    remaining = set(sd)
    loaded = unexpected = 0
    for f in sorted(Path(snapshot_path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                mapped = _map_vision_key(key)
                if mapped is None:
                    continue
                if mapped not in sd:
                    unexpected += 1
                    continue
                t = sf.get_tensor(key)
                with torch.no_grad():
                    sd[mapped].copy_(t.to(sd[mapped].dtype).to(sd[mapped].device))
                loaded += 1
                remaining.discard(mapped)
    print(f"      loaded={loaded} missing={len(remaining)} unexpected_vision_keys={unexpected}")
    if remaining:
        print(f"      first 10 missing: {sorted(remaining)[:10]}")
        sys.exit(2)

    print(f"[4/5] Prepare pixel_values + grid_thw via HF processor ...")
    img = _synthetic_image()
    inputs = processor(text=["<|vision_start|><|image_pad|><|vision_end|>"], images=[img], return_tensors="pt")
    pixel_values = inputs["pixel_values"].cuda()
    grid_thw = inputs["image_grid_thw"].cuda()
    print(f"      pixel_values.shape={tuple(pixel_values.shape)} grid_thw={grid_thw.tolist()}")

    print(f"[5/5] Run both vision towers and compare ...")
    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        snapshot_path, torch_dtype=dtype, trust_remote_code=True
    ).cuda().eval()

    with torch.no_grad():
        hf_out = hf_model.model.visual(pixel_values.to(hf_model.model.visual.dtype), grid_thw=grid_thw)
        hf_pooler = hf_out.pooler_output
        hf_last = hf_out.last_hidden_state
        hf_ds = hf_out.deepstack_features

        our_pool, our_last, our_ds = our_vt(pixel_values.to(dtype), grid_thw)

    def diff(a: torch.Tensor, b: torch.Tensor, name: str) -> None:
        d = (a.float() - b.float()).abs()
        print(f"  {name:30s} shape={tuple(a.shape)}  max={d.max().item():.4f}  mean={d.mean().item():.6f}")

    print(f"\n[compare]")
    diff(our_last, hf_last, "last_hidden_state")
    diff(our_pool, hf_pooler, "pooler_output")
    for i, (ours, hfs) in enumerate(zip(our_ds, hf_ds)):
        diff(ours, hfs, f"deepstack_features[{i}]")

    summarize_fallbacks()


if __name__ == "__main__":
    main()
