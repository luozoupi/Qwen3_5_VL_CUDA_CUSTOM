#include "common.h"

namespace {
template <typename scalar_t>
__global__ void residual_add_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    int64_t n_elements) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    out[idx] = a[idx] + b[idx];
}
}  // namespace

torch::Tensor residual_add_forward_cuda(const torch::Tensor& a, const torch::Tensor& b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    TORCH_CHECK(a.sizes() == b.sizes(), "residual_add shapes must match");
    c10::cuda::CUDAGuard guard(a.device());

    auto out = torch::empty_like(a);
    const int64_t n = a.numel();
    if (n == 0) return out;

    const int threads = 256;
    const int blocks = static_cast<int>(CEIL_DIV(n, threads));
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(a.scalar_type(), "residual_add", [&] {
        residual_add_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
