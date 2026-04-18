#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> topk_forward_cuda(const torch::Tensor& x, int64_t k);
torch::Tensor index_add_forward_cuda(
    torch::Tensor target, const torch::Tensor& source, const torch::Tensor& index);
torch::Tensor batched_gemm_forward_cuda(const torch::Tensor& x, const torch::Tensor& weight);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("topk_forward(Tensor x, int k) -> Tensor[]");
    m.def("index_add_forward(Tensor(a!) target, Tensor source, Tensor index) -> Tensor(a!)");
    m.def("batched_gemm_forward(Tensor x, Tensor weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("topk_forward", &topk_forward_cuda);
    m.impl("index_add_forward", &index_add_forward_cuda);
    m.impl("batched_gemm_forward", &batched_gemm_forward_cuda);
}
