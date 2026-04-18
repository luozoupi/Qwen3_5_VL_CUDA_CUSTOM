#include "common.h"

// GELU tanh approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

namespace {
constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluC = 0.044715f;

template <typename scalar_t>
__global__ void gelu_tanh_fwd_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = static_cast<float>(x[idx]);
    const float v3 = v * v * v;
    const float inner = kSqrt2OverPi * (v + kGeluC * v3);
    const float t = tanhf(inner);
    out[idx] = static_cast<scalar_t>(0.5f * v * (1.0f + t));
}

template <typename scalar_t>
__global__ void gelu_tanh_bwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_x,
    int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = static_cast<float>(x[idx]);
    const float v2 = v * v;
    const float inner = kSqrt2OverPi * (v + kGeluC * v * v2);
    const float t = tanhf(inner);
    const float sech2 = 1.0f - t * t;
    const float dinner = kSqrt2OverPi * (1.0f + 3.0f * kGeluC * v2);
    const float dy = 0.5f * (1.0f + t) + 0.5f * v * sech2 * dinner;
    grad_x[idx] = static_cast<scalar_t>(static_cast<float>(grad_out[idx]) * dy);
}
}  // namespace

torch::Tensor gelu_tanh_forward_cuda(const torch::Tensor& x) {
    CHECK_INPUT(x);
    c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x);
    const int64_t n = x.numel();
    if (n == 0) return out;
    const int threads = 256;
    const int blocks = static_cast<int>(CEIL_DIV(n, threads));
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(x.scalar_type(), "gelu_tanh_fwd", [&] {
        gelu_tanh_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gelu_tanh_backward_cuda(const torch::Tensor& x, const torch::Tensor& grad_out) {
    CHECK_INPUT(x);
    CHECK_INPUT(grad_out);
    c10::cuda::CUDAGuard guard(x.device());
    auto grad_x = torch::empty_like(x);
    const int64_t n = x.numel();
    if (n == 0) return grad_x;
    const int threads = 256;
    const int blocks = static_cast<int>(CEIL_DIV(n, threads));
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(x.scalar_type(), "gelu_tanh_bwd", [&] {
        gelu_tanh_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(), grad_x.data_ptr<scalar_t>(), n);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}
