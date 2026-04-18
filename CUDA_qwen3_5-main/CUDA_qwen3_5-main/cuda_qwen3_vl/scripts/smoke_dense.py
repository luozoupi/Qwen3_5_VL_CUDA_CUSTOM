"""Text-only smoke test for CudaQwen3VLDenseModel against HF Qwen3-VL-8B-Instruct.

Loads real HF weights via our mapper, runs a forward pass on a short prompt, and
(optionally) compares logits to the HF reference model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from cuda_qwen3_vl.configs import Qwen3VLConfig
from cuda_qwen3_vl.kernels import summarize_fallbacks
from cuda_qwen3_vl.loaders import load_hf_weights
from cuda_qwen3_vl.models import CudaQwen3VLDenseModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Text-only smoke for CudaQwen3VLDenseModel")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compare-hf", action="store_true", help="Load HF reference and compare logits")
    parser.add_argument("--load-only", action="store_true", help="Only test weight loading, no forward")
    parser.add_argument("--list-mismatches", type=int, default=20,
                        help="How many missing/unexpected keys to print")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"[1/5] Downloading {args.model_id} ...")
    snapshot_path = snapshot_download(
        repo_id=args.model_id,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
    )
    print(f"      -> {snapshot_path}")

    print(f"[2/5] Loading HF config and normalizing ...")
    from transformers import AutoConfig, AutoTokenizer
    hf_cfg = AutoConfig.from_pretrained(snapshot_path, trust_remote_code=True)
    cfg = Qwen3VLConfig.from_hf_config(hf_cfg)
    print(f"      family={cfg.family} hidden={cfg.text.hidden_size} layers={cfg.text.num_layers} "
          f"vocab={cfg.text.vocab_size} vision_hidden={cfg.vision.hidden_size}")
    tok = AutoTokenizer.from_pretrained(snapshot_path, trust_remote_code=True)

    if cfg.family != "dense":
        print(f"[!] family is {cfg.family}, not dense. Aborting.")
        sys.exit(1)

    print(f"[3/5] Instantiating CudaQwen3VLDenseModel (this allocates {args.dtype} weights on CUDA) ...")
    torch.cuda.empty_cache()
    model = CudaQwen3VLDenseModel(cfg).to(dtype=dtype).cuda().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      total params: {n_params/1e6:.1f}M")

    print(f"[4/5] Loading HF weights via cuda_qwen3_vl loader ...")
    report = load_hf_weights(model, snapshot_path)
    print(f"      loaded: {len(report['loaded'])}")
    print(f"      missing: {len(report['missing'])}")
    print(f"      unexpected: {len(report['unexpected'])}")
    print(f"      mismatched: {len(report['mismatched'])}")
    if report["missing"]:
        print(f"      first {args.list_mismatches} missing: {report['missing'][:args.list_mismatches]}")
    if report["unexpected"]:
        print(f"      first {args.list_mismatches} unexpected: {report['unexpected'][:args.list_mismatches]}")
    if report["mismatched"]:
        print(f"      first {args.list_mismatches} mismatched: {report['mismatched'][:args.list_mismatches]}")

    if args.load_only:
        summarize_fallbacks()
        return

    if report["missing"] or report["mismatched"]:
        print("[!] Load was not clean — aborting forward (would give nonsense). "
              "Fix the loader name mapping first.")
        summarize_fallbacks()
        sys.exit(2)

    print(f"[5/5] Running text-only forward: {args.prompt!r}")
    input_ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    print(f"      input_ids.shape={tuple(input_ids.shape)}")
    with torch.no_grad():
        logits = model(input_ids=input_ids)
    print(f"      logits.shape={tuple(logits.shape)} "
          f"range=[{logits.min().item():.3f}, {logits.max().item():.3f}] "
          f"finite={torch.isfinite(logits).all().item()}")

    next_token = logits[0, -1].argmax(dim=-1).item()
    print(f"      argmax next token: id={next_token} text={tok.decode([next_token])!r}")

    if args.compare_hf:
        print(f"\n[compare] Loading HF reference model ...")
        from transformers import Qwen3VLForConditionalGeneration
        hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
            snapshot_path, torch_dtype=dtype, trust_remote_code=True
        ).cuda().eval()
        with torch.no_grad():
            hf_out = hf_model(input_ids=input_ids)
        hf_logits = hf_out.logits
        abs_diff = (logits.float() - hf_logits.float()).abs()
        print(f"      logit_max_abs_diff={abs_diff.max().item():.4f}")
        print(f"      logit_mean_abs_diff={abs_diff.mean().item():.6f}")
        hf_next = hf_logits[0, -1].argmax(dim=-1).item()
        print(f"      HF argmax: id={hf_next} text={tok.decode([hf_next])!r}")
        match = next_token == hf_next
        print(f"      next-token match: {match}")

    summarize_fallbacks()


if __name__ == "__main__":
    main()
