#include <torch/extension.h>

torch::Tensor gelu_tanh_forward_cuda(const torch::Tensor& x);
torch::Tensor gelu_tanh_backward_cuda(const torch::Tensor& x, const torch::Tensor& grad_out);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("gelu_tanh_forward(Tensor x) -> Tensor");
    m.def("gelu_tanh_backward(Tensor x, Tensor grad_out) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("gelu_tanh_forward", &gelu_tanh_forward_cuda);
    m.impl("gelu_tanh_backward", &gelu_tanh_backward_cuda);
}
