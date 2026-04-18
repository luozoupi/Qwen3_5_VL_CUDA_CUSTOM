"""KV-cache generation smoke: our CUDA dense model generates N tokens greedily.

Compares the generated sequence to HF's .generate() with the same prompt, same
do_sample=False, same max_new_tokens.
"""
from __future__ import annotations

import argparse
import time

import torch
from huggingface_hub import snapshot_download

from cuda_qwen3_vl.configs import Qwen3VLConfig
from cuda_qwen3_vl.kernels import summarize_fallbacks
from cuda_qwen3_vl.loaders import load_hf_weights
from cuda_qwen3_vl.models import CudaQwen3VLDenseModel


def main() -> None:
    parser = argparse.ArgumentParser(description="KV-cache generation smoke")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compare-hf", action="store_true")
    args = parser.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"[1/4] Snapshot {args.model_id} ...")
    snapshot_path = snapshot_download(
        repo_id=args.model_id,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
    )

    print(f"[2/4] Load config + tokenizer ...")
    from transformers import AutoConfig, AutoTokenizer
    hf_cfg = AutoConfig.from_pretrained(snapshot_path, trust_remote_code=True)
    cfg = Qwen3VLConfig.from_hf_config(hf_cfg)
    tok = AutoTokenizer.from_pretrained(snapshot_path, trust_remote_code=True)

    print(f"[3/4] Build + load our CUDA model ...")
    torch.cuda.empty_cache()
    model = CudaQwen3VLDenseModel(cfg).to(dtype=dtype).cuda().eval()
    report = load_hf_weights(model, snapshot_path)
    print(f"      loaded={len(report['loaded'])} missing={len(report['missing'])}")

    input_ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    prompt_len = input_ids.shape[1]
    print(f"      prompt tokens: {prompt_len}")

    print(f"[4/4] Greedy generate {args.max_new_tokens} tokens with KV-cache ...")
    t0 = time.time()
    out = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
    torch.cuda.synchronize()
    dt = time.time() - t0
    new_tokens = out[0, prompt_len:].tolist()
    print(f"      elapsed: {dt:.2f}s ({args.max_new_tokens/dt:.2f} tok/s)")
    print(f"      our generated: {new_tokens}")
    print(f"      our text:    {tok.decode(out[0].tolist())!r}")

    if args.compare_hf:
        print(f"\n[compare] HF greedy generate ...")
        from transformers import Qwen3VLForConditionalGeneration
        hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
            snapshot_path, torch_dtype=dtype, trust_remote_code=True
        ).cuda().eval()
        t0 = time.time()
        hf_out = hf_model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        dt = time.time() - t0
        hf_new = hf_out[0, prompt_len:].tolist()
        print(f"      hf elapsed: {dt:.2f}s ({args.max_new_tokens/dt:.2f} tok/s)")
        print(f"      hf generated: {hf_new}")
        print(f"      hf text:    {tok.decode(hf_out[0].tolist())!r}")

        n_match = sum(1 for a, b in zip(new_tokens, hf_new) if a == b)
        print(f"\n      matching tokens (left-to-right): {n_match}/{min(len(new_tokens), len(hf_new))}")

    summarize_fallbacks()


if __name__ == "__main__":
    main()
