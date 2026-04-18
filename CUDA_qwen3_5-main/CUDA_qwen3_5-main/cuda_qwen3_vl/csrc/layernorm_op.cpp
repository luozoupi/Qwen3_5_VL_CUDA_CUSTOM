#include <torch/extension.h>

torch::Tensor layernorm_forward_cuda(
    const torch::Tensor& x, const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias, double eps);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("layernorm_forward(Tensor x, Tensor weight, Tensor? bias, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("layernorm_forward", &layernorm_forward_cuda);
}
