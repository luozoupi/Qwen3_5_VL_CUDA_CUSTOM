#include <torch/extension.h>

torch::Tensor conv3d_patch_forward_cuda(
    const torch::Tensor& input, const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("conv3d_patch_forward(Tensor input, Tensor weight, Tensor? bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("conv3d_patch_forward", &conv3d_patch_forward_cuda);
}
