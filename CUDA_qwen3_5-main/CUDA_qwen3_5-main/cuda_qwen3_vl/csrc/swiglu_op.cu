#include "common.h"

// SwiGLU: out = silu(gate) * up, where silu(x) = x * sigmoid(x)

namespace {
template <typename scalar_t>
__global__ void swiglu_fwd_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    scalar_t* __restrict__ out,
    int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g = static_cast<float>(gate[idx]);
    const float u = static_cast<float>(up[idx]);
    const float s = 1.0f / (1.0f + __expf(-g));
    out[idx] = static_cast<scalar_t>(g * s * u);
}

template <typename scalar_t>
__global__ void swiglu_bwd_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_gate,
    scalar_t* __restrict__ grad_up,
    int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g = static_cast<float>(gate[idx]);
    const float u = static_cast<float>(up[idx]);
    const float go = static_cast<float>(grad_out[idx]);
    const float sig = 1.0f / (1.0f + __expf(-g));
    const float silu = g * sig;
    // d/dg silu(g) = sig + g*sig*(1-sig) = sig*(1 + g*(1-sig))
    const float dsilu_dg = sig * (1.0f + g * (1.0f - sig));
    grad_gate[idx] = static_cast<scalar_t>(go * u * dsilu_dg);
    grad_up[idx] = static_cast<scalar_t>(go * silu);
}
}  // namespace

torch::Tensor swiglu_forward_cuda(const torch::Tensor& gate, const torch::Tensor& up) {
    CHECK_INPUT(gate);
    CHECK_INPUT(up);
    c10::cuda::CUDAGuard guard(gate.device());
    auto out = torch::empty_like(gate);
    const int64_t n = gate.numel();
    if (n == 0) return out;
    const int threads = 256;
    const int blocks = static_cast<int>(CEIL_DIV(n, threads));
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(gate.scalar_type(), "swiglu_fwd", [&] {
        swiglu_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> swiglu_backward_cuda(
    const torch::Tensor& gate, const torch::Tensor& up, const torch::Tensor& grad_out) {
    CHECK_INPUT(gate);
    CHECK_INPUT(up);
    CHECK_INPUT(grad_out);
    c10::cuda::CUDAGuard guard(gate.device());
    auto grad_gate = torch::empty_like(gate);
    auto grad_up = torch::empty_like(up);
    const int64_t n = gate.numel();
    if (n == 0) return {grad_gate, grad_up};
    const int threads = 256;
    const int blocks = static_cast<int>(CEIL_DIV(n, threads));
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(gate.scalar_type(), "swiglu_bwd", [&] {
        swiglu_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(),
            grad_gate.data_ptr<scalar_t>(), grad_up.data_ptr<scalar_t>(), n);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_gate, grad_up};
}
