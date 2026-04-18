#include "common.h"

// Standard interleaved RoPE over the last dim of Q/K.
// x shape: (B, H, S, D). cos/sin shape: (B, S, D_rope) or (S, D_rope).
// Rotates pairs: x[..., :D/2] and x[..., D/2:D] via cos/sin.

namespace {
template <typename scalar_t>
__global__ void rope_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    scalar_t* __restrict__ out,
    int64_t B, int64_t H, int64_t S, int64_t D, int64_t D_rope,
    int64_t cos_batch_stride) {
    const int64_t row = blockIdx.x;
    if (row >= B * H * S) return;
    const int64_t s = row % S;
    const int64_t b = (row / (H * S));
    const scalar_t* x_row = x + row * D;
    scalar_t* o_row = out + row * D;
    const scalar_t* cos_row = cos + b * cos_batch_stride + s * D_rope;
    const scalar_t* sin_row = sin + b * cos_batch_stride + s * D_rope;

    const int half = D_rope / 2;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        if (i < half) {
            const float x1 = static_cast<float>(x_row[i]);
            const float x2 = static_cast<float>(x_row[i + half]);
            const float c = static_cast<float>(cos_row[i]);
            const float sn = static_cast<float>(sin_row[i]);
            o_row[i] = static_cast<scalar_t>(x1 * c - x2 * sn);
        } else if (i < D_rope) {
            const int j = i - half;
            const float x1 = static_cast<float>(x_row[j]);
            const float x2 = static_cast<float>(x_row[i]);
            const float c = static_cast<float>(cos_row[j]);
            const float sn = static_cast<float>(sin_row[j]);
            o_row[i] = static_cast<scalar_t>(x2 * c + x1 * sn);
        } else {
            // pass-through for non-rotated tail
            o_row[i] = x_row[i];
        }
    }
}
}  // namespace

torch::Tensor rope_forward_cuda(
    const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin) {
    CHECK_INPUT(x);
    CHECK_INPUT(cos);
    CHECK_INPUT(sin);
    TORCH_CHECK(x.dim() == 4, "x must be (B, H, S, D)");
    c10::cuda::CUDAGuard guard(x.device());

    const int64_t B = x.size(0);
    const int64_t H = x.size(1);
    const int64_t S = x.size(2);
    const int64_t D = x.size(3);
    const int64_t D_rope = cos.size(-1);

    // cos shape may be (S, D_rope) or (B, S, D_rope)
    int64_t cos_batch_stride = 0;
    if (cos.dim() == 3) cos_batch_stride = cos.stride(0);

    auto out = torch::empty_like(x);
    const int threads = std::min<int64_t>(256, D);
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(x.scalar_type(), "rope_fwd", [&] {
        rope_fwd_kernel<scalar_t><<<static_cast<int>(B * H * S), threads, 0, stream>>>(
            x.data_ptr<scalar_t>(), cos.data_ptr<scalar_t>(), sin.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), B, H, S, D, D_rope, cos_batch_stride);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
