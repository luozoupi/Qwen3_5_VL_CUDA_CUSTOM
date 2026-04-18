#include "common.h"

namespace {
template <typename scalar_t>
__global__ void embedding_fwd_kernel(
    const int64_t* __restrict__ ids,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    int64_t n_tokens, int64_t embed_dim) {
    const int token = blockIdx.x;
    if (token >= n_tokens) return;
    const int64_t id = ids[token];
    const scalar_t* w_row = weight + id * embed_dim;
    scalar_t* o_row = out + token * embed_dim;
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        o_row[i] = w_row[i];
    }
}

template <typename scalar_t>
__global__ void embedding_bwd_kernel(
    const int64_t* __restrict__ ids,
    const scalar_t* __restrict__ grad_out,
    float* __restrict__ grad_weight,  // fp32 accumulator
    int64_t n_tokens, int64_t embed_dim, int64_t padding_idx) {
    const int token = blockIdx.x;
    if (token >= n_tokens) return;
    const int64_t id = ids[token];
    if (id == padding_idx) return;
    const scalar_t* g_row = grad_out + token * embed_dim;
    float* gw_row = grad_weight + id * embed_dim;
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        atomicAdd(&gw_row[i], static_cast<float>(g_row[i]));
    }
}
}  // namespace

torch::Tensor embedding_forward_cuda(const torch::Tensor& ids, const torch::Tensor& weight) {
    CHECK_INPUT(ids);
    CHECK_INPUT(weight);
    TORCH_CHECK(ids.scalar_type() == torch::kInt64, "ids must be int64");
    c10::cuda::CUDAGuard guard(weight.device());

    const auto ids_shape = ids.sizes().vec();
    auto ids_flat = ids.reshape({-1});
    const int64_t n_tokens = ids_flat.size(0);
    const int64_t embed_dim = weight.size(1);

    auto out = torch::empty({n_tokens, embed_dim}, weight.options());
    if (n_tokens == 0) {
        auto out_shape = ids_shape;
        out_shape.push_back(embed_dim);
        return out.reshape(out_shape);
    }

    const int threads = std::min<int64_t>(256, embed_dim);
    const int blocks = static_cast<int>(n_tokens);
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(weight.scalar_type(), "embedding_fwd", [&] {
        embedding_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            ids_flat.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), n_tokens, embed_dim);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto out_shape = ids_shape;
    out_shape.push_back(embed_dim);
    return out.reshape(out_shape);
}

torch::Tensor embedding_backward_cuda(
    const torch::Tensor& ids, const torch::Tensor& grad_out,
    int64_t num_embeddings, int64_t padding_idx) {
    CHECK_INPUT(ids);
    CHECK_INPUT(grad_out);
    c10::cuda::CUDAGuard guard(grad_out.device());

    auto ids_flat = ids.reshape({-1});
    auto go_2d = grad_out.reshape({-1, grad_out.size(-1)});
    const int64_t n_tokens = ids_flat.size(0);
    const int64_t embed_dim = go_2d.size(1);

    auto grad_weight = torch::zeros({num_embeddings, embed_dim},
                                    grad_out.options().dtype(torch::kFloat32));
    if (n_tokens == 0) return grad_weight.to(grad_out.dtype());

    const int threads = std::min<int64_t>(256, embed_dim);
    const int blocks = static_cast<int>(n_tokens);
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(grad_out.scalar_type(), "embedding_bwd", [&] {
        embedding_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            ids_flat.data_ptr<int64_t>(), go_2d.data_ptr<scalar_t>(),
            grad_weight.data_ptr<float>(), n_tokens, embed_dim, padding_idx);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_weight.to(grad_out.dtype());
}
