#include <torch/extension.h>

torch::Tensor softmax_forward_cuda(const torch::Tensor& x);
torch::Tensor softmax_backward_cuda(const torch::Tensor& out, const torch::Tensor& grad_out);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("softmax_forward(Tensor x) -> Tensor");
    m.def("softmax_backward(Tensor out, Tensor grad_out) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("softmax_forward", &softmax_forward_cuda);
    m.impl("softmax_backward", &softmax_backward_cuda);
}
