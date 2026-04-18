"""Pre-compile all CUDA extensions so subsequent runs load from cache."""
from __future__ import annotations

import argparse
import sys

from cuda_qwen3_vl.kernels._loader import load_op, get_load_error, summarize_fallbacks


ALL_OPS = [
    "residual_add",
    "sigmoid_mul",
    "gelu_tanh",
    "swiglu",
    "rmsnorm",
    "layernorm",
    "embedding",
    "softmax",
    "cross_entropy",
    "rope",
    "mrope",
    "matmul",
    "flash_attention",
    "moe_routing",
    "conv3d_patch",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all cuda_qwen3_vl CUDA extensions")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    failures = []
    for op in ALL_OPS:
        print(f"[build] {op} ... ", end="", flush=True)
        ok = load_op(op, verbose=args.verbose)
        if ok:
            print("OK")
        else:
            err = get_load_error(op)
            print(f"FAIL: {type(err).__name__}: {err}")
            failures.append(op)

    print()
    summarize_fallbacks()
    if failures:
        print(f"\n{len(failures)} op(s) failed to build: {failures}", file=sys.stderr)
        sys.exit(1)
    print(f"\nAll {len(ALL_OPS)} ops built successfully.")


if __name__ == "__main__":
    main()
