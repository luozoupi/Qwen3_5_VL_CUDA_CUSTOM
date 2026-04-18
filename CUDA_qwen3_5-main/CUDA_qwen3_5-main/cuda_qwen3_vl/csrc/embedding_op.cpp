#include <torch/extension.h>

torch::Tensor embedding_forward_cuda(const torch::Tensor& ids, const torch::Tensor& weight);
torch::Tensor embedding_backward_cuda(
    const torch::Tensor& ids, const torch::Tensor& grad_out,
    int64_t num_embeddings, int64_t padding_idx);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("embedding_forward(Tensor ids, Tensor weight) -> Tensor");
    m.def("embedding_backward(Tensor ids, Tensor grad_out, int num_embeddings, int padding_idx) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("embedding_forward", &embedding_forward_cuda);
    m.impl("embedding_backward", &embedding_backward_cuda);
}
