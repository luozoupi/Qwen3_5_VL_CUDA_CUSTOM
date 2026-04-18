"""MoE smoke: load Qwen3-VL-30B-A3B weights, run text-only forward, compare to HF."""
from __future__ import annotations

import argparse
import sys

import torch
from huggingface_hub import snapshot_download

from cuda_qwen3_vl.configs import Qwen3VLConfig
from cuda_qwen3_vl.kernels import summarize_fallbacks
from cuda_qwen3_vl.loaders import load_hf_weights
from cuda_qwen3_vl.models import CudaQwen3VLMoeModel


def main() -> None:
    parser = argparse.ArgumentParser(description="MoE smoke for Qwen3-VL-30B-A3B")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compare-hf", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    args = parser.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"[1/5] Download checkpoint ({args.model_id}) ...")
    snapshot_path = snapshot_download(
        repo_id=args.model_id,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
    )
    print(f"      -> {snapshot_path}")

    print(f"[2/5] Load HF config and normalize to MoE layout ...")
    from transformers import AutoConfig, AutoTokenizer
    hf_cfg = AutoConfig.from_pretrained(snapshot_path, trust_remote_code=True)
    cfg = Qwen3VLConfig.from_hf_config(hf_cfg)
    tok = AutoTokenizer.from_pretrained(snapshot_path, trust_remote_code=True)
    print(f"      family={cfg.family} hidden={cfg.text.hidden_size} layers={cfg.text.num_layers} "
          f"experts={getattr(cfg.text, 'num_experts', None)} topk={getattr(cfg.text, 'num_experts_per_tok', None)} "
          f"moe_intermediate={getattr(cfg.text, 'moe_intermediate_size', None)}")
    if cfg.family != "moe":
        print(f"[!] family is {cfg.family}, not moe. Aborting.")
        sys.exit(1)

    print(f"[3/5] Instantiate CudaQwen3VLMoeModel (allocating {args.dtype} weights on CUDA) ...")
    torch.cuda.empty_cache()
    model = CudaQwen3VLMoeModel(cfg).to(dtype=dtype).cuda().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      total params: {n_params/1e9:.2f}B")

    print(f"[4/5] Load HF weights via loader ...")
    report = load_hf_weights(model, snapshot_path)
    print(f"      loaded={len(report['loaded'])} missing={len(report['missing'])} "
          f"unexpected={len(report['unexpected'])} mismatched={len(report['mismatched'])}")
    if report["missing"]:
        print(f"      first 10 missing: {report['missing'][:10]}")
    if report["unexpected"]:
        print(f"      first 10 unexpected: {report['unexpected'][:10]}")
    if report["mismatched"]:
        print(f"      first 5 mismatched: {report['mismatched'][:5]}")

    if args.load_only or report["missing"] or report["mismatched"]:
        if report["missing"] or report["mismatched"]:
            print("[!] load not clean — skip forward")
        summarize_fallbacks()
        return

    print(f"[5/5] Run text-only forward: {args.prompt!r}")
    input_ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        logits, router_logits = model(input_ids=input_ids)
    print(f"      logits.shape={tuple(logits.shape)} finite={torch.isfinite(logits).all().item()} "
          f"range=[{logits.min().item():.3f}, {logits.max().item():.3f}]")
    next_token = logits[0, -1].argmax(-1).item()
    print(f"      argmax next: id={next_token} text={tok.decode([next_token])!r}")
    print(f"      router_logits: {len(router_logits)} layers — sample shape {tuple(router_logits[0].shape) if router_logits else 'N/A'}")

    if args.compare_hf:
        print("\n[compare] Loading HF reference ...")
        from transformers import Qwen3VLMoeForConditionalGeneration
        hf_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            snapshot_path, torch_dtype=dtype, trust_remote_code=True
        ).cuda().eval()
        with torch.no_grad():
            hf_out = hf_model(input_ids=input_ids)
        hf_next = hf_out.logits[0, -1].argmax(-1).item()
        abs_diff = (logits.float() - hf_out.logits.float()).abs()
        print(f"      logit_max_abs_diff={abs_diff.max().item():.4f} "
              f"logit_mean_abs_diff={abs_diff.mean().item():.6f}")
        print(f"      our argmax={next_token} ({tok.decode([next_token])!r})  "
              f"hf argmax={hf_next} ({tok.decode([hf_next])!r})  match={next_token == hf_next}")
        our_top5 = logits[0, -1].topk(5).indices.tolist()
        hf_top5 = hf_out.logits[0, -1].topk(5).indices.tolist()
        print(f"      our top-5: {our_top5}  hf top-5: {hf_top5}  "
              f"overlap={len(set(our_top5) & set(hf_top5))}/5")

    summarize_fallbacks()


if __name__ == "__main__":
    main()
