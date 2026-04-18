# CUDA Qwen3-VL

Hand-written CUDA kernels replacing **every replaceable operator** — including GEMM — for two HuggingFace models:

- **Qwen3-VL (dense)** — `Qwen3VLForConditionalGeneration` (e.g. 8B). Vision tower + dense LLM.
- **Qwen3-VL-MoE** — `Qwen3VLMoeForConditionalGeneration` (e.g. 30B-A3B). Same vision tower + MoE LLM.

Companion to [Qwen-Triton](../Qwen-Triton), but uses C++/CUDA via `torch.utils.cpp_extension.load` JIT compilation instead of Triton.

Debug env: conda `py310_2`, GPU 4.

## Layout

```
CUDA_qwen3_5/
  cuda_qwen3_vl/
    csrc/                   # All CUDA kernels (.cpp + .cu)
    kernels/                # Python JIT loaders + autograd wrappers
    modules/                # nn.Module drop-ins (CudaLinear, CudaVisionBlock, CudaSparseMoE, ...)
    models/                 # Vision tower + dense / MoE model assemblies
    configs/                # HF config normalization
    loaders/                # HF safetensors weight loading + name remapping
    scripts/                # build_all_ops, smoke tests
  tests/                    # 31 tests: kernel parity + module sanity + model forward
```

Shared `csrc/` and `kernels/` → per-model classes in `models/dense.py` and `models/moe.py` share the same kernel + module layer. The vision tower is identical between the two.

## Kernel Inventory (15 CUDA files)

| Kernel | File | Forward | Backward |
|---|---|:-:|:-:|
| residual_add | `csrc/residual_add_op.*` | ✓ | identity (in autograd wrapper) |
| sigmoid_mul | `csrc/sigmoid_mul_op.*` | ✓ | ✓ |
| gelu_tanh | `csrc/gelu_tanh_op.*` | ✓ | ✓ |
| swiglu (silu-mul) | `csrc/swiglu_op.*` | ✓ | ✓ |
| rmsnorm | `csrc/rmsnorm_op.*` | ✓ | ✓ |
| layernorm | `csrc/layernorm_op.*` | ✓ | torch.autograd via ref |
| embedding | `csrc/embedding_op.*` | ✓ | ✓ (atomic) |
| softmax | `csrc/softmax_op.*` | ✓ | ✓ |
| cross_entropy | `csrc/cross_entropy_op.*` | ✓ | ✓ |
| rope (2D vision) | `csrc/rope_op.*` | ✓ | ✓ (via neg-sin trick) |
| mrope (3D text) | `csrc/mrope_op.*` | ✓ | ✓ (via neg-sin trick) |
| matmul (GEMM) | `csrc/matmul_op.*` | ✓ | ✓ |
| flash_attention | `csrc/flash_attention_op.*` | ✓ (causal + GQA) | torch.autograd via SDPA ref |
| moe_routing | `csrc/moe_routing_op.*` | topk, index_add, batched_gemm | — |
| conv3d_patch | `csrc/conv3d_patch_op.*` | ✓ | — |

## Module Layer (drop-in for `nn.Linear` / `nn.Embedding` / etc.)

- `CudaLinear(in, out, bias)` — same `weight`/`bias` names/shapes as `nn.Linear` → `load_state_dict` works unchanged.
- `CudaEmbedding(num, dim, padding_idx)` — same `weight` as `nn.Embedding`.
- `CudaRMSNorm`, `CudaLayerNorm` — same parameter layout.
- `CudaVisionAttention`, `CudaFullAttention` (causal+GQA), `CudaVisionMLP`, `CudaSwiGLUMLP`, `CudaSparseMoE`.
- `CudaVisionBlock`, `CudaTextDecoderLayer`, `CudaVisionPatchEmbed`, `CudaVisionPatchMerger`.

## Models

- `CudaQwen3VLDenseModel` — dense Qwen3-VL: `CudaVisionTower` + embed + decoder stack (SwiGLU MLP) + lm_head
- `CudaQwen3VLMoeModel` — MoE Qwen3-VL: same skeleton, decoder MLPs are `CudaSparseMoE` per config rules

Both take `(input_ids, pixel_values, vision_position_ids, image_token_mask, position_ids)` and return logits (and router logits for MoE). Vision features are scattered into positions marked by `image_token_mask`.

## Quick Start

### Build all 15 kernels (JIT compile on first run, cached thereafter)

```bash
CUDA_VISIBLE_DEVICES=4 python -m cuda_qwen3_vl.scripts.build_all_ops
```

### Run the full test suite (31 tests)

```bash
CUDA_VISIBLE_DEVICES=4 python -m pytest tests/ -v
```

### Import and use

```python
import torch
from cuda_qwen3_vl.configs import Qwen3VLConfig
from cuda_qwen3_vl.models import CudaQwen3VLDenseModel
from cuda_qwen3_vl.loaders import load_hf_weights

# Load a real Qwen3-VL checkpoint
from transformers import AutoConfig
hf_cfg = AutoConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
cfg = Qwen3VLConfig.from_hf_config(hf_cfg)

model = CudaQwen3VLDenseModel(cfg).cuda().eval()
report = load_hf_weights(model, snapshot_path="/path/to/hf/snapshot")
print(f"loaded={len(report['loaded'])} missing={len(report['missing'])}")
```

## Build & ABI Details

The JIT loader (`kernels/_loader.py`) does three non-obvious things:

1. **Uses system `gcc-11` / `g++-11`** as compiler (via `CC`/`CXX`) instead of the conda env's default. Reason: py310_2's default PATH picks up py310's GCC 14 which requires `CXXABI_1.3.15` in libstdc++ — a symbol py310_2's libstdc++ doesn't provide. GCC 11 produces code that runs against the system's libstdc++ without mismatch.
2. **Detects libstdc++ ABI** at import time and, if the currently-loaded libstdc++ is too old, auto re-executes the process with `LD_PRELOAD` pointing at a newer one. Disable via `CUDA_QWEN3_VL_AUTO_PRELOAD=0`.
3. **Adds RPATH** at link time pointing at the newer libstdc++ dir as a belt-and-suspenders fallback.

All three are logged with `[CUDA-QWEN3-VL-FALLBACK]` prefix to stderr.

## Fallback Visibility

Every fallback (build failure, runtime exception, or deliberate unimplemented path) is recorded in `FALLBACK_LOG` and tagged `[CUDA-QWEN3-VL-FALLBACK]` in stderr. Call `summarize_fallbacks()` at end-of-run for a report:

```python
from cuda_qwen3_vl.kernels import summarize_fallbacks
summarize_fallbacks()
# [CUDA-QWEN3-VL-FALLBACK] no fallbacks recorded — all CUDA kernels executed.
# or, on failure:
# [CUDA-QWEN3-VL-FALLBACK] === FALLBACK SUMMARY (2 events) ===
#   flash_attention: 1 fallback(s)
#     - backward_not_implemented_uses_sdpa_autograd
```

Enable strict mode (`CUDA_QWEN3_VL_STRICT=1`) to turn any runtime fallback into a hard error.

**Known fallbacks** (deliberate, not bugs):
- `layernorm.backward` — torch autograd on reference impl (CUDA bwd not yet implemented)
- `flash_attention.backward` — torch autograd via SDPA (CUDA bwd not yet implemented)
- `conv3d_patch.backward` — not yet implemented (vision tower inference path only)

## Known Limitations

- **Custom GEMM is slower than cuBLAS** (expected trade-off for "every op in custom CUDA"). Toggle `CUDA_QWEN3_VL_USE_CUBLAS=1` to route `matmul` through `F.linear` for perf comparison.
- Flash Attention block sizes fixed at `BM=16, BN=32` to fit shared memory on Ada/Hopper; head_dim up to 128 supported.
- MoE forward only (backward would need gradient of `batched_gemm` and `cuda_topk` w.r.t. gate).
- Real Qwen3-VL-8B forward parity vs HF reference is the intended minimum experiment; MoE 30B does not fit on a single GPU in bf16, so tiny-config parity is the bound.

## Test Summary

- **22 kernel parity tests** — every kernel's fwd (and bwd where implemented) compared to PyTorch ref (atol≈1e-3 bf16, 1e-4 fp32)
- **7 module sanity tests** — drop-in parity for CudaLinear/CudaEmbedding/CudaRMSNorm/CudaLayerNorm and forward-finite for vision block, SwiGLU MLP, MoE block
- **2 end-to-end model tests** — dense + MoE tiny-config forward produces finite logits

Run: `CUDA_VISIBLE_DEVICES=4 pytest tests/ -v`
