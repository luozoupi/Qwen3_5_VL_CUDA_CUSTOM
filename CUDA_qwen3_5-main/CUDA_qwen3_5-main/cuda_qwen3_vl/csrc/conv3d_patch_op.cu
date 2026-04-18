#include "common.h"

// Conv3d patch embed with non-overlapping kernel (stride == kernel_size).
// Special-cased for Qwen3-VL: in_channels=3, kernel=[temporal, H_patch, W_patch].
// Input: (N_patches, C, T, H, W) where T=temporal_patch_size, H=W=patch_size
// Weight: (embed_dim, C, T, H, W) — but we lower to GEMM: im2col → 3x3x patch → matmul.
// Output: (N_patches, embed_dim)
//
// Implementation: each output row is a dot-product of weight-per-output-channel
// with the flattened patch vector. So: output[n, e] = sum_{c,t,h,w} input[n, c, t, h, w] * weight[e, c, t, h, w] + bias[e]
// This is just: flatten patch to (N, C*T*H*W), and weight to (embed_dim, C*T*H*W), then matmul.
// We expose a direct elementwise-matmul kernel here.

namespace {
template <typename scalar_t>
__global__ void conv3d_patch_kernel(
    const scalar_t* __restrict__ input,   // (N, C*T*H*W) flat
    const scalar_t* __restrict__ weight,  // (embed_dim, C*T*H*W) flat
    const scalar_t* __restrict__ bias,    // (embed_dim,) or nullptr
    scalar_t* __restrict__ out,           // (N, embed_dim)
    int64_t N, int64_t E, int64_t K) {
    const int n = blockIdx.x;
    const int e = blockIdx.y * blockDim.x + threadIdx.x;
    if (n >= N || e >= E) return;
    const scalar_t* x_row = input + n * K;
    const scalar_t* w_row = weight + e * K;
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) acc += static_cast<float>(x_row[i]) * static_cast<float>(w_row[i]);
    if (bias != nullptr) acc += static_cast<float>(bias[e]);
    out[n * E + e] = static_cast<scalar_t>(acc);
}
}  // namespace

torch::Tensor conv3d_patch_forward_cuda(
    const torch::Tensor& input,   // (N, C, T, H, W)
    const torch::Tensor& weight,  // (E, C, T, H, W)
    const c10::optional<torch::Tensor>& bias) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    TORCH_CHECK(input.dim() == 5, "input must be 5D (N,C,T,H,W)");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D (E,C,T,H,W)");
    c10::cuda::CUDAGuard guard(input.device());

    const int64_t N = input.size(0);
    const int64_t E = weight.size(0);
    const int64_t K = input.size(1) * input.size(2) * input.size(3) * input.size(4);

    auto input_flat = input.reshape({N, K});
    auto weight_flat = weight.reshape({E, K});

    auto out = torch::empty({N, E}, input.options());
    if (N == 0 || E == 0) return out;

    const int threads = 32;
    const dim3 grid((unsigned)N, (unsigned)CEIL_DIV(E, threads));
    const dim3 block(threads);
    const auto stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor bias_tensor;
    const bool has_bias = bias.has_value();
    if (has_bias) bias_tensor = bias->contiguous();

    DISPATCH_FLOAT_TYPES(input.scalar_type(), "conv3d_patch", [&] {
        const scalar_t* bias_ptr = has_bias ? bias_tensor.data_ptr<scalar_t>() : nullptr;
        conv3d_patch_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input_flat.data_ptr<scalar_t>(), weight_flat.data_ptr<scalar_t>(),
            bias_ptr, out.data_ptr<scalar_t>(), N, E, K);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
