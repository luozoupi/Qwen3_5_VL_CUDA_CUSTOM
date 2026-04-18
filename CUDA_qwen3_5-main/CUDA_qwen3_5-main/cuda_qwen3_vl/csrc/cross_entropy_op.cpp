#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> cross_entropy_forward_cuda(
    const torch::Tensor& logits, const torch::Tensor& targets, int64_t ignore_index);
torch::Tensor cross_entropy_backward_cuda(
    const torch::Tensor& logits, const torch::Tensor& targets,
    const torch::Tensor& lse, double grad_scale, int64_t ignore_index);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("cross_entropy_forward(Tensor logits, Tensor targets, int ignore_index) -> Tensor[]");
    m.def("cross_entropy_backward(Tensor logits, Tensor targets, Tensor lse, float grad_scale, int ignore_index) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("cross_entropy_forward", &cross_entropy_forward_cuda);
    m.impl("cross_entropy_backward", &cross_entropy_backward_cuda);
}
