#include "common.h"

// LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias

namespace {
template <typename scalar_t>
__global__ void layernorm_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t n_rows, int64_t n_cols, float eps) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;
    const int blk = blockDim.x;
    const scalar_t* x_row = x + row * n_cols;
    scalar_t* out_row = out + row * n_cols;

    float sum = 0.0f, sumsq = 0.0f;
    for (int i = tid; i < n_cols; i += blk) {
        const float v = static_cast<float>(x_row[i]);
        sum += v;
        sumsq += v * v;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    }
    __shared__ float s_sum[32];
    __shared__ float s_sumsq[32];
    const int lane = tid % 32;
    const int warp = tid / 32;
    if (lane == 0) { s_sum[warp] = sum; s_sumsq[warp] = sumsq; }
    __syncthreads();
    if (warp == 0) {
        sum = (tid < (blk + 31) / 32) ? s_sum[lane] : 0.0f;
        sumsq = (tid < (blk + 31) / 32) ? s_sumsq[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
            sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
        }
        if (lane == 0) { s_sum[0] = sum; s_sumsq[0] = sumsq; }
    }
    __syncthreads();

    const float mean = s_sum[0] / static_cast<float>(n_cols);
    const float var = s_sumsq[0] / static_cast<float>(n_cols) - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < n_cols; i += blk) {
        const float v = static_cast<float>(x_row[i]);
        const float w = static_cast<float>(weight[i]);
        const float b = bias != nullptr ? static_cast<float>(bias[i]) : 0.0f;
        out_row[i] = static_cast<scalar_t>((v - mean) * inv_std * w + b);
    }
}
}  // namespace

torch::Tensor layernorm_forward_cuda(
    const torch::Tensor& x, const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias, double eps) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    c10::cuda::CUDAGuard guard(x.device());

    const auto shape = x.sizes().vec();
    const int64_t n_cols = shape.back();
    auto x2d = x.reshape({-1, n_cols});
    auto out = torch::empty_like(x2d);
    const int64_t n_rows = x2d.size(0);

    const int threads = std::min<int64_t>(1024, ((n_cols + 31) / 32) * 32);
    const int blocks = static_cast<int>(n_rows);
    const auto stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor bias_tensor;
    const bool has_bias = bias.has_value();
    if (has_bias) bias_tensor = bias->contiguous();

    DISPATCH_FLOAT_TYPES(x.scalar_type(), "layernorm_fwd", [&] {
        const scalar_t* bias_ptr = has_bias ? bias_tensor.data_ptr<scalar_t>() : nullptr;
        layernorm_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x2d.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
            bias_ptr, out.data_ptr<scalar_t>(),
            n_rows, n_cols, static_cast<float>(eps));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.reshape(shape);
}
