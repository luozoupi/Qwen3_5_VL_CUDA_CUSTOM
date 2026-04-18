"""Image+text smoke for CudaQwen3VLDenseModel vs HF Qwen3VLForConditionalGeneration.

Uses HF's processor to prepare inputs, HF's vision tower to compute features +
deepstack embeds (since our vision tower hasn't been parity-validated yet),
and runs our CUDA text stack with the correct deepstack fusion. Compares final
logits to HF reference.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image

from cuda_qwen3_vl.configs import Qwen3VLConfig
from cuda_qwen3_vl.kernels import summarize_fallbacks
from cuda_qwen3_vl.loaders import load_hf_weights
from cuda_qwen3_vl.models import CudaQwen3VLDenseModel


def _get_sample_image() -> Image.Image:
    """Synthetic 224x224 image: red/green/blue/yellow quadrants with a diagonal stripe.

    Deterministic and network-free. HF processor will resize/normalize as needed.
    """
    from PIL import ImageDraw
    img = Image.new("RGB", (224, 224), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 112, 112], fill=(220, 30, 30))
    d.rectangle([112, 0, 224, 112], fill=(30, 180, 30))
    d.rectangle([0, 112, 112, 224], fill=(30, 30, 200))
    d.rectangle([112, 112, 224, 224], fill=(230, 210, 40))
    d.line([(0, 0), (223, 223)], fill="black", width=6)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Image+text smoke for CudaQwen3VLDenseModel")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--image", default=None, help="Local image path; if omitted, downloads a sample")
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--use-cuda-vision", action="store_true",
                        help="Use our CudaVisionTower instead of HF's (vision-tower parity check).")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"[1/7] Download checkpoint snapshot for {args.model_id} ...")
    snapshot_path = snapshot_download(
        repo_id=args.model_id,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"],
    )
    print(f"      -> {snapshot_path}")

    print(f"[2/7] Load HF processor + config ...")
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    hf_cfg = AutoConfig.from_pretrained(snapshot_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(snapshot_path, trust_remote_code=True)
    cfg = Qwen3VLConfig.from_hf_config(hf_cfg)
    print(f"      family={cfg.family} text_hidden={cfg.text.hidden_size} layers={cfg.text.num_layers} "
          f"vision_hidden={cfg.vision.hidden_size} deepstack_layers={cfg.vision.deepstack_layers}")

    print(f"[3/7] Build and load our CUDA model weights ...")
    torch.cuda.empty_cache()
    our_model = CudaQwen3VLDenseModel(cfg).to(dtype=dtype).cuda().eval()
    report = load_hf_weights(our_model, snapshot_path)
    print(f"      loaded={len(report['loaded'])} missing={len(report['missing'])} "
          f"unexpected={len(report['unexpected'])}")
    if report["missing"]:
        print(f"      [!] missing keys (first 10): {report['missing'][:10]}")
        sys.exit(2)

    print(f"[4/7] Prepare image + prompt via HF processor ...")
    image = Image.open(args.image).convert("RGB") if args.image else _get_sample_image()
    print(f"      image size: {image.size}")
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": args.prompt}],
    }]
    batch_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[batch_text], images=[image], padding=True, return_tensors="pt",
    )
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
    print(f"      input_ids.shape={tuple(inputs['input_ids'].shape)} "
          f"pixel_values.shape={tuple(inputs['pixel_values'].shape)} "
          f"image_grid_thw={inputs['image_grid_thw'].tolist()} "
          f"extra_keys={sorted(set(inputs.keys()) - {'input_ids','pixel_values','image_grid_thw','attention_mask'})}")

    print(f"[5/7] Load HF reference model (vision + text ref) ...")
    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        snapshot_path, torch_dtype=dtype, trust_remote_code=True
    ).cuda().eval()

    print(f"[6/7] Extract vision features + deepstack ({'our CudaVisionTower' if args.use_cuda_vision else 'HF vision tower'}) ...")
    with torch.no_grad():
        pixel_values = inputs["pixel_values"].to(hf_model.model.visual.dtype)
        grid_thw = inputs["image_grid_thw"]
        if args.use_cuda_vision:
            vision_feats, _last, deepstack_feats = our_model.visual(pixel_values.to(dtype), grid_thw)
        else:
            vision_out = hf_model.model.visual(pixel_values, grid_thw=grid_thw)
            vision_feats = vision_out.pooler_output
            deepstack_feats = vision_out.deepstack_features
        print(f"      vision_feats.shape={tuple(vision_feats.shape)} "
              f"deepstack_n={len(deepstack_feats) if deepstack_feats else 0}")

        # Build inputs_embeds: scatter vision_feats into image-token positions
        image_token_id = hf_cfg.image_token_id if hasattr(hf_cfg, "image_token_id") else \
                         processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        input_ids = inputs["input_ids"]
        image_mask = (input_ids == image_token_id)
        print(f"      image_token_id={image_token_id} image_mask.sum()={image_mask.sum().item()}")
        assert image_mask.sum() == vision_feats.shape[0], \
            f"image token count {image_mask.sum().item()} != vision_feats.shape[0]={vision_feats.shape[0]}"

        inputs_embeds = our_model.embed_tokens(input_ids).to(dtype)
        inputs_embeds_scattered = inputs_embeds.clone()
        inputs_embeds_scattered[image_mask] = vision_feats.to(dtype)

        # Compute 4-row position_ids using HF's get_rope_index
        # mm_token_type_ids comes from the processor; required for M-RoPE with images.
        mm_tti = inputs.get("mm_token_type_ids")
        if mm_tti is None:
            raise RuntimeError("processor output missing mm_token_type_ids — cannot compute M-RoPE")
        position_ids, _rope_deltas = hf_model.model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_tti,
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
            attention_mask=inputs.get("attention_mask"),
        )
        print(f"      position_ids.shape={tuple(position_ids.shape)}")

    print(f"[7/7] Run our CUDA text stack with vision fusion ...")
    with torch.no_grad():
        our_logits = our_model(
            inputs_embeds=inputs_embeds_scattered,
            position_ids=position_ids,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=[d.to(dtype) for d in deepstack_feats] if deepstack_feats else None,
        )
    print(f"      our_logits.shape={tuple(our_logits.shape)} "
          f"range=[{our_logits.min().item():.3f}, {our_logits.max().item():.3f}] "
          f"finite={torch.isfinite(our_logits).all().item()}")

    with torch.no_grad():
        hf_out = hf_model(**inputs)
    hf_logits = hf_out.logits
    abs_diff = (our_logits.float() - hf_logits.float()).abs()
    print(f"\n[compare] logit_max_abs_diff={abs_diff.max().item():.4f} "
          f"logit_mean_abs_diff={abs_diff.mean().item():.6f}")

    our_next = our_logits[0, -1].argmax(-1).item()
    hf_next = hf_logits[0, -1].argmax(-1).item()
    print(f"[compare] argmax our={our_next} ({processor.tokenizer.decode([our_next])!r})  "
          f"hf={hf_next} ({processor.tokenizer.decode([hf_next])!r})  match={our_next == hf_next}")

    # Top-5 agreement as a softer signal
    our_top5 = our_logits[0, -1].topk(5).indices.tolist()
    hf_top5 = hf_logits[0, -1].topk(5).indices.tolist()
    print(f"[compare] our top-5: {our_top5}  hf top-5: {hf_top5}  "
          f"overlap={len(set(our_top5) & set(hf_top5))}/5")

    summarize_fallbacks()


if __name__ == "__main__":
    main()
