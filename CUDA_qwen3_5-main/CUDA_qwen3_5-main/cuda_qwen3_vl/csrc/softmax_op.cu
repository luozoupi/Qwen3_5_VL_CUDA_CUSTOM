#include "common.h"
#include <cfloat>

// Row-wise softmax over last dim.

namespace {
template <typename scalar_t>
__global__ void softmax_fwd_kernel(
    const scalar_t* __restrict__ x, scalar_t* __restrict__ out,
    int64_t n_rows, int64_t n_cols) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;
    const int blk = blockDim.x;
    const scalar_t* x_row = x + row * n_cols;
    scalar_t* o_row = out + row * n_cols;

    // Pass 1: max
    float m = -FLT_MAX;
    for (int i = tid; i < n_cols; i += blk) m = fmaxf(m, static_cast<float>(x_row[i]));
    for (int off = 16; off > 0; off /= 2) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
    __shared__ float s_m[32];
    const int lane = tid % 32, warp = tid / 32;
    if (lane == 0) s_m[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = (tid < (blk + 31) / 32) ? s_m[lane] : -FLT_MAX;
        for (int off = 16; off > 0; off /= 2) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
        if (lane == 0) s_m[0] = m;
    }
    __syncthreads();
    const float row_max = s_m[0];

    // Pass 2: sum of exp
    float s = 0.0f;
    for (int i = tid; i < n_cols; i += blk) s += __expf(static_cast<float>(x_row[i]) - row_max);
    for (int off = 16; off > 0; off /= 2) s += __shfl_down_sync(0xffffffff, s, off);
    __shared__ float s_s[32];
    if (lane == 0) s_s[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = (tid < (blk + 31) / 32) ? s_s[lane] : 0.0f;
        for (int off = 16; off > 0; off /= 2) s += __shfl_down_sync(0xffffffff, s, off);
        if (lane == 0) s_s[0] = s;
    }
    __syncthreads();
    const float inv_sum = 1.0f / s_s[0];

    for (int i = tid; i < n_cols; i += blk) {
        const float v = __expf(static_cast<float>(x_row[i]) - row_max) * inv_sum;
        o_row[i] = static_cast<scalar_t>(v);
    }
}

template <typename scalar_t>
__global__ void softmax_bwd_kernel(
    const scalar_t* __restrict__ out, const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_x, int64_t n_rows, int64_t n_cols) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;
    const int blk = blockDim.x;
    const scalar_t* o_row = out + row * n_cols;
    const scalar_t* go_row = grad_out + row * n_cols;
    scalar_t* gx_row = grad_x + row * n_cols;

    float dot = 0.0f;
    for (int i = tid; i < n_cols; i += blk)
        dot += static_cast<float>(o_row[i]) * static_cast<float>(go_row[i]);
    for (int off = 16; off > 0; off /= 2) dot += __shfl_down_sync(0xffffffff, dot, off);
    __shared__ float s_dot[32];
    const int lane = tid % 32, warp = tid / 32;
    if (lane == 0) s_dot[warp] = dot;
    __syncthreads();
    if (warp == 0) {
        dot = (tid < (blk + 31) / 32) ? s_dot[lane] : 0.0f;
        for (int off = 16; off > 0; off /= 2) dot += __shfl_down_sync(0xffffffff, dot, off);
        if (lane == 0) s_dot[0] = dot;
    }
    __syncthreads();
    const float d = s_dot[0];

    for (int i = tid; i < n_cols; i += blk) {
        const float o = static_cast<float>(o_row[i]);
        const float g = static_cast<float>(go_row[i]);
        gx_row[i] = static_cast<scalar_t>(o * (g - d));
    }
}
}  // namespace

torch::Tensor softmax_forward_cuda(const torch::Tensor& x) {
    CHECK_INPUT(x);
    c10::cuda::CUDAGuard guard(x.device());
    const auto shape = x.sizes().vec();
    const int64_t n_cols = shape.back();
    auto x2d = x.reshape({-1, n_cols});
    auto out = torch::empty_like(x2d);
    const int64_t n_rows = x2d.size(0);
    const int threads = std::min<int64_t>(1024, ((n_cols + 31) / 32) * 32);
    const int blocks = static_cast<int>(n_rows);
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(x.scalar_type(), "softmax_fwd", [&] {
        softmax_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x2d.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n_rows, n_cols);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.reshape(shape);
}

torch::Tensor softmax_backward_cuda(const torch::Tensor& out, const torch::Tensor& grad_out) {
    CHECK_INPUT(out);
    CHECK_INPUT(grad_out);
    c10::cuda::CUDAGuard guard(out.device());
    const auto shape = out.sizes().vec();
    const int64_t n_cols = shape.back();
    auto out2d = out.reshape({-1, n_cols});
    auto go2d = grad_out.reshape({-1, n_cols});
    auto grad_x = torch::empty_like(out2d);
    const int64_t n_rows = out2d.size(0);
    const int threads = std::min<int64_t>(1024, ((n_cols + 31) / 32) * 32);
    const int blocks = static_cast<int>(n_rows);
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(out.scalar_type(), "softmax_bwd", [&] {
        softmax_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out2d.data_ptr<scalar_t>(), go2d.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(), n_rows, n_cols);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x.reshape(shape);
}
