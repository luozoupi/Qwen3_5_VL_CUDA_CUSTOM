#include <torch/extension.h>
#include <vector>

torch::Tensor matmul_forward_cuda(
    const torch::Tensor& x, const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias);
std::vector<torch::Tensor> matmul_backward_cuda(
    const torch::Tensor& x, const torch::Tensor& weight,
    const torch::Tensor& grad_out, bool needs_bias_grad);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("matmul_forward(Tensor x, Tensor weight, Tensor? bias) -> Tensor");
    m.def("matmul_backward(Tensor x, Tensor weight, Tensor grad_out, bool needs_bias_grad) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("matmul_forward", &matmul_forward_cuda);
    m.impl("matmul_backward", &matmul_backward_cuda);
}
