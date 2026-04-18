#include <torch/extension.h>
#include <vector>

torch::Tensor rmsnorm_forward_cuda(const torch::Tensor& x, const torch::Tensor& weight, double eps);
std::vector<torch::Tensor> rmsnorm_backward_cuda(
    const torch::Tensor& x, const torch::Tensor& weight,
    const torch::Tensor& grad_out, double eps);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("rmsnorm_forward(Tensor x, Tensor weight, float eps) -> Tensor");
    m.def("rmsnorm_backward(Tensor x, Tensor weight, Tensor grad_out, float eps) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("rmsnorm_forward", &rmsnorm_forward_cuda);
    m.impl("rmsnorm_backward", &rmsnorm_backward_cuda);
}
