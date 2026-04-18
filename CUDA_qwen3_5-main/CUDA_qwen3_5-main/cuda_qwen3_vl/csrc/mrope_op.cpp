#include <torch/extension.h>

torch::Tensor mrope_forward_cuda(const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("mrope_forward(Tensor x, Tensor cos, Tensor sin) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("mrope_forward", &mrope_forward_cuda);
}
