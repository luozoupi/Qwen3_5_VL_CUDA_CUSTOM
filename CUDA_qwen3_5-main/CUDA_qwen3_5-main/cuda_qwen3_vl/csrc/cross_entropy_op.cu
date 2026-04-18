#include "common.h"
#include <cfloat>

// Cross-entropy per-token with ignore_index handling.
// Forward: returns per-token loss + logsumexp.
// Backward: returns grad_logits = (softmax - one_hot(target)) * upstream_grad / n_valid.

namespace {
template <typename scalar_t>
__global__ void ce_fwd_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ loss_per_token,
    float* __restrict__ lse,
    int64_t N, int64_t V, int64_t ignore_index) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const int blk = blockDim.x;
    const int64_t tgt = targets[row];
    if (tgt == ignore_index) {
        if (tid == 0) { loss_per_token[row] = 0.0f; lse[row] = 0.0f; }
        return;
    }
    const scalar_t* x_row = logits + row * V;

    // Pass 1: max
    float m = -FLT_MAX;
    for (int i = tid; i < V; i += blk) m = fmaxf(m, static_cast<float>(x_row[i]));
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
    for (int i = tid; i < V; i += blk) s += __expf(static_cast<float>(x_row[i]) - row_max);
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
    if (tid == 0) {
        const float log_sum = row_max + logf(s_s[0]);
        lse[row] = log_sum;
        loss_per_token[row] = log_sum - static_cast<float>(x_row[tgt]);
    }
}

template <typename scalar_t>
__global__ void ce_bwd_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ lse,
    scalar_t* __restrict__ grad_logits,
    float grad_scale, int64_t N, int64_t V, int64_t ignore_index) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const int blk = blockDim.x;
    const int64_t tgt = targets[row];
    const scalar_t* x_row = logits + row * V;
    scalar_t* g_row = grad_logits + row * V;

    if (tgt == ignore_index) {
        for (int i = tid; i < V; i += blk) g_row[i] = static_cast<scalar_t>(0.0f);
        return;
    }

    const float log_sum = lse[row];
    for (int i = tid; i < V; i += blk) {
        const float p = __expf(static_cast<float>(x_row[i]) - log_sum);
        const float is_tgt = (i == tgt) ? 1.0f : 0.0f;
        g_row[i] = static_cast<scalar_t>((p - is_tgt) * grad_scale);
    }
}
}  // namespace

std::vector<torch::Tensor> cross_entropy_forward_cuda(
    const torch::Tensor& logits, const torch::Tensor& targets, int64_t ignore_index) {
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);
    TORCH_CHECK(targets.scalar_type() == torch::kInt64, "targets must be int64");
    c10::cuda::CUDAGuard guard(logits.device());

    const int64_t N = logits.size(0);
    const int64_t V = logits.size(1);
    auto loss_per_token = torch::empty({N}, logits.options().dtype(torch::kFloat32));
    auto lse = torch::empty({N}, logits.options().dtype(torch::kFloat32));

    if (N > 0) {
        const int threads = std::min<int64_t>(1024, ((V + 31) / 32) * 32);
        const auto stream = at::cuda::getCurrentCUDAStream();
        DISPATCH_FLOAT_TYPES(logits.scalar_type(), "ce_fwd", [&] {
            ce_fwd_kernel<scalar_t><<<static_cast<int>(N), threads, 0, stream>>>(
                logits.data_ptr<scalar_t>(), targets.data_ptr<int64_t>(),
                loss_per_token.data_ptr<float>(), lse.data_ptr<float>(),
                N, V, ignore_index);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {loss_per_token, lse};
}

torch::Tensor cross_entropy_backward_cuda(
    const torch::Tensor& logits, const torch::Tensor& targets,
    const torch::Tensor& lse, double grad_scale, int64_t ignore_index) {
    CHECK_INPUT(logits);
    c10::cuda::CUDAGuard guard(logits.device());
    const int64_t N = logits.size(0);
    const int64_t V = logits.size(1);
    auto grad_logits = torch::empty_like(logits);
    if (N > 0) {
        const int threads = std::min<int64_t>(1024, ((V + 31) / 32) * 32);
        const auto stream = at::cuda::getCurrentCUDAStream();
        DISPATCH_FLOAT_TYPES(logits.scalar_type(), "ce_bwd", [&] {
            ce_bwd_kernel<scalar_t><<<static_cast<int>(N), threads, 0, stream>>>(
                logits.data_ptr<scalar_t>(), targets.data_ptr<int64_t>(),
                lse.data_ptr<float>(), grad_logits.data_ptr<scalar_t>(),
                static_cast<float>(grad_scale), N, V, ignore_index);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return grad_logits;
}
