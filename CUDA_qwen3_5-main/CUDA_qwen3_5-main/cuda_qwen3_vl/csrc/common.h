#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMacros.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define DISPATCH_FLOAT_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, TYPE, NAME, __VA_ARGS__)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__device__ __forceinline__ int64_t next_pow2(int64_t x) {
    int64_t p = 1;
    while (p < x) p <<= 1;
    return p;
}
