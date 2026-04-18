#include "common.h"
#include <cuda_runtime.h>

// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
// Block-per-row, threads perform row-reduction.

namespace {
template <typename scalar_t>
__global__ void rmsnorm_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    int64_t n_rows,
    int64_t n_cols,
    float eps) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;
    const int blk = blockDim.x;

    const scalar_t* x_row = x + row * n_cols;
    scalar_t* out_row = out + row * n_cols;

    // Reduce sum of squares
    float sumsq = 0.0f;
    for (int i = tid; i < n_cols; i += blk) {
        const float v = static_cast<float>(x_row[i]);
        sumsq += v * v;
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    }

    // Block reduce across warps
    __shared__ float block_sumsq[32];
    const int lane = tid % 32;
    const int warp = tid / 32;
    if (lane == 0) block_sumsq[warp] = sumsq;
    __syncthreads();

    if (warp == 0) {
        sumsq = (tid < (blk + 31) / 32) ? block_sumsq[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
        }
        if (lane == 0) block_sumsq[0] = sumsq;
    }
    __syncthreads();

    const float variance = block_sumsq[0] / static_cast<float>(n_cols);
    const float inv_std = rsqrtf(variance + eps);

    for (int i = tid; i < n_cols; i += blk) {
        const float v = static_cast<float>(x_row[i]);
        const float w = static_cast<float>(weight[i]);
        out_row[i] = static_cast<scalar_t>(v * inv_std * w);
    }
}

template <typename scalar_t>
__global__ void rmsnorm_bwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_x,
    float* __restrict__ grad_weight_partial,  // (num_blocks, n_cols), reduced on host
    int64_t n_rows,
    int64_t n_cols,
    float eps) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;
    const int blk = blockDim.x;

    const scalar_t* x_row = x + row * n_cols;
    const scalar_t* go_row = grad_out + row * n_cols;
    scalar_t* gx_row = grad_x + row * n_cols;

    // Two-pass reduction: sum(x^2) and sum(grad_out * weight * x)
    float sumsq = 0.0f;
    float dot = 0.0f;
    for (int i = tid; i < n_cols; i += blk) {
        const float v = static_cast<float>(x_row[i]);
        const float w = static_cast<float>(weight[i]);
        const float g = static_cast<float>(go_row[i]);
        sumsq += v * v;
        dot += g * w * v;
    }

    // Block reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    }
    __shared__ float block_sumsq[32];
    __shared__ float block_dot[32];
    const int lane = tid % 32;
    const int warp = tid / 32;
    if (lane == 0) { block_sumsq[warp] = sumsq; block_dot[warp] = dot; }
    __syncthreads();
    if (warp == 0) {
        sumsq = (tid < (blk + 31) / 32) ? block_sumsq[lane] : 0.0f;
        dot = (tid < (blk + 31) / 32) ? block_dot[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        if (lane == 0) { block_sumsq[0] = sumsq; block_dot[0] = dot; }
    }
    __syncthreads();

    const float variance = block_sumsq[0] / static_cast<float>(n_cols);
    const float inv_std = rsqrtf(variance + eps);
    const float inv_cols = 1.0f / static_cast<float>(n_cols);
    const float dot_term = block_dot[0] * inv_cols;
    const float inv_std_cubed = inv_std * inv_std * inv_std;

    for (int i = tid; i < n_cols; i += blk) {
        const float v = static_cast<float>(x_row[i]);
        const float w = static_cast<float>(weight[i]);
        const float g = static_cast<float>(go_row[i]);
        // grad_x = g * w * inv_std - v * inv_std^3 * dot_term
        const float gx = g * w * inv_std - v * inv_std_cubed * dot_term;
        gx_row[i] = static_cast<scalar_t>(gx);
        // grad_weight contribution = g * v * inv_std (atomic into row-0 partial)
        atomicAdd(&grad_weight_partial[i], g * v * inv_std);
    }
}
}  // namespace

torch::Tensor rmsnorm_forward_cuda(const torch::Tensor& x, const torch::Tensor& weight, double eps) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    c10::cuda::CUDAGuard guard(x.device());

    const auto original_shape = x.sizes().vec();
    const int64_t n_cols = original_shape.back();
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == n_cols, "weight shape mismatch");

    auto x2d = x.reshape({-1, n_cols});
    auto out = torch::empty_like(x2d);
    const int64_t n_rows = x2d.size(0);

    const int threads = std::min<int64_t>(1024, ((n_cols + 31) / 32) * 32);
    const int blocks = static_cast<int>(n_rows);
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(x.scalar_type(), "rmsnorm_fwd", [&] {
        rmsnorm_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x2d.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
            n_rows, n_cols, static_cast<float>(eps));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.reshape(original_shape);
}

std::vector<torch::Tensor> rmsnorm_backward_cuda(
    const torch::Tensor& x, const torch::Tensor& weight,
    const torch::Tensor& grad_out, double eps) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    CHECK_INPUT(grad_out);
    c10::cuda::CUDAGuard guard(x.device());

    const auto original_shape = x.sizes().vec();
    const int64_t n_cols = original_shape.back();

    auto x2d = x.reshape({-1, n_cols});
    auto go2d = grad_out.reshape({-1, n_cols});
    auto grad_x = torch::empty_like(x2d);
    auto grad_weight = torch::zeros({n_cols}, x.options().dtype(torch::kFloat32));
    const int64_t n_rows = x2d.size(0);

    const int threads = std::min<int64_t>(1024, ((n_cols + 31) / 32) * 32);
    const int blocks = static_cast<int>(n_rows);
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(x.scalar_type(), "rmsnorm_bwd", [&] {
        rmsnorm_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x2d.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), go2d.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(), grad_weight.data_ptr<float>(),
            n_rows, n_cols, static_cast<float>(eps));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_x.reshape(original_shape), grad_weight.to(weight.dtype())};
}
