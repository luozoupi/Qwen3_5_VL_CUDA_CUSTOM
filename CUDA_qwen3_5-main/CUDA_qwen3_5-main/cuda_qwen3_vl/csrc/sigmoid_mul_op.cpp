#include <torch/extension.h>
#include <vector>

torch::Tensor sigmoid_mul_forward_cuda(const torch::Tensor& x, const torch::Tensor& gate);
std::vector<torch::Tensor> sigmoid_mul_backward_cuda(
    const torch::Tensor& x, const torch::Tensor& gate, const torch::Tensor& grad_out);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("sigmoid_mul_forward(Tensor x, Tensor gate) -> Tensor");
    m.def("sigmoid_mul_backward(Tensor x, Tensor gate, Tensor grad_out) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("sigmoid_mul_forward", &sigmoid_mul_forward_cuda);
    m.impl("sigmoid_mul_backward", &sigmoid_mul_backward_cuda);
}
