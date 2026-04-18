#include <torch/extension.h>
#include <vector>

torch::Tensor swiglu_forward_cuda(const torch::Tensor& gate, const torch::Tensor& up);
std::vector<torch::Tensor> swiglu_backward_cuda(
    const torch::Tensor& gate, const torch::Tensor& up, const torch::Tensor& grad_out);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("swiglu_forward(Tensor gate, Tensor up) -> Tensor");
    m.def("swiglu_backward(Tensor gate, Tensor up, Tensor grad_out) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("swiglu_forward", &swiglu_forward_cuda);
    m.impl("swiglu_backward", &swiglu_backward_cuda);
}
