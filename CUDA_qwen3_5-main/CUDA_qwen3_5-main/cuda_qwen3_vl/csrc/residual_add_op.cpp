#include <torch/extension.h>

torch::Tensor residual_add_forward_cuda(const torch::Tensor& a, const torch::Tensor& b);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("residual_add_forward(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("residual_add_forward", &residual_add_forward_cuda);
}
