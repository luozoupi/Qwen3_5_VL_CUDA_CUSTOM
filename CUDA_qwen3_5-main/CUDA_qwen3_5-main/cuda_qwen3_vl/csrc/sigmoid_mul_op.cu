#include "common.h"

namespace {
template <typename scalar_t>
__global__ void sigmoid_mul_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gate,
    scalar_t* __restrict__ out,
    int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g = static_cast<float>(gate[idx]);
    const float s = 1.0f / (1.0f + __expf(-g));
    out[idx] = static_cast<scalar_t>(static_cast<float>(x[idx]) * s);
}

template <typename scalar_t>
__global__ void sigmoid_mul_bwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_gate,
    int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float xv = static_cast<float>(x[idx]);
    const float gv = static_cast<float>(gate[idx]);
    const float go = static_cast<float>(grad_out[idx]);
    const float s = 1.0f / (1.0f + __expf(-gv));
    grad_x[idx] = static_cast<scalar_t>(go * s);
    grad_gate[idx] = static_cast<scalar_t>(go * xv * s * (1.0f - s));
}
}  // namespace

torch::Tensor sigmoid_mul_forward_cuda(const torch::Tensor& x, const torch::Tensor& gate) {
    CHECK_INPUT(x);
    CHECK_INPUT(gate);
    c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x);
    const int64_t n = x.numel();
    if (n == 0) return out;
    const int threads = 256;
    const int blocks = static_cast<int>(CEIL_DIV(n, threads));
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(x.scalar_type(), "sigmoid_mul_fwd", [&] {
        sigmoid_mul_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> sigmoid_mul_backward_cuda(
    const torch::Tensor& x, const torch::Tensor& gate, const torch::Tensor& grad_out) {
    CHECK_INPUT(x);
    CHECK_INPUT(gate);
    CHECK_INPUT(grad_out);
    c10::cuda::CUDAGuard guard(x.device());
    auto grad_x = torch::empty_like(x);
    auto grad_gate = torch::empty_like(gate);
    const int64_t n = x.numel();
    if (n == 0) return {grad_x, grad_gate};
    const int threads = 256;
    const int blocks = static_cast<int>(CEIL_DIV(n, threads));
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(x.scalar_type(), "sigmoid_mul_bwd", [&] {
        sigmoid_mul_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(), grad_gate.data_ptr<scalar_t>(), n);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_x, grad_gate};
}
