#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> flash_attention_forward_cuda(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    double scale, bool is_causal, int64_t num_kv_groups);

std::vector<torch::Tensor> flash_attention_backward_cuda(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& o, const torch::Tensor& lse, const torch::Tensor& grad_o,
    double scale, bool is_causal, int64_t num_kv_groups);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("flash_attention_forward(Tensor q, Tensor k, Tensor v, float scale, bool is_causal, int num_kv_groups) -> Tensor[]");
    m.def("flash_attention_backward(Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, Tensor grad_o, float scale, bool is_causal, int num_kv_groups) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("flash_attention_forward", &flash_attention_forward_cuda);
    m.impl("flash_attention_backward", &flash_attention_backward_cuda);
}
