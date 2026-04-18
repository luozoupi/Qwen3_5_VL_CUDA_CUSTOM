#include <torch/extension.h>

torch::Tensor rope_forward_cuda(const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("rope_forward(Tensor x, Tensor cos, Tensor sin) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("rope_forward", &rope_forward_cuda);
}
