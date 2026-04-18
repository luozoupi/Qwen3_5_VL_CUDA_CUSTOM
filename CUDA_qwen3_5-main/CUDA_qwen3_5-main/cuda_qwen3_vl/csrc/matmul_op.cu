#include "common.h"

// Tiled GEMM: Y = X @ W^T + bias.
// X: (M, K), W: (N, K), Y: (M, N). Matches nn.Linear layout.
// Uses shared memory tiles, fp32 accumulation.

namespace {
constexpr int kBM = 32;
constexpr int kBN = 32;
constexpr int kBK = 32;
constexpr int kTM = 4;  // Each thread computes 4x4 output tile
constexpr int kTN = 4;
constexpr int kThreadsPerBlock = (kBM / kTM) * (kBN / kTN);  // 8x8 = 64

template <typename scalar_t>
__global__ void matmul_fwd_kernel(
    const scalar_t* __restrict__ X,   // (M, K)
    const scalar_t* __restrict__ W,   // (N, K)
    const scalar_t* __restrict__ bias,  // (N,) or nullptr
    scalar_t* __restrict__ Y,         // (M, N)
    int64_t M, int64_t N, int64_t K) {
    __shared__ float A[kBM][kBK];
    __shared__ float B[kBN][kBK];  // B[n][k] = W[n_tile + n, k_tile + k]

    const int bx = blockIdx.x;  // N tile
    const int by = blockIdx.y;  // M tile
    const int tid = threadIdx.x;
    const int threads = blockDim.x;

    const int m_tile = by * kBM;
    const int n_tile = bx * kBN;

    float acc[kTM][kTN];
    #pragma unroll
    for (int i = 0; i < kTM; ++i)
        #pragma unroll
        for (int j = 0; j < kTN; ++j) acc[i][j] = 0.0f;

    const int thread_row = tid / (kBN / kTN);  // 0..(kBM/kTM-1)
    const int thread_col = tid % (kBN / kTN);

    for (int k_tile = 0; k_tile < K; k_tile += kBK) {
        // Load X tile (kBM x kBK) cooperatively
        for (int i = tid; i < kBM * kBK; i += threads) {
            const int r = i / kBK;
            const int c = i % kBK;
            const int64_t gm = m_tile + r;
            const int64_t gk = k_tile + c;
            A[r][c] = (gm < M && gk < K) ? static_cast<float>(X[gm * K + gk]) : 0.0f;
        }
        // Load W tile (kBN x kBK) cooperatively
        for (int i = tid; i < kBN * kBK; i += threads) {
            const int r = i / kBK;
            const int c = i % kBK;
            const int64_t gn = n_tile + r;
            const int64_t gk = k_tile + c;
            B[r][c] = (gn < N && gk < K) ? static_cast<float>(W[gn * K + gk]) : 0.0f;
        }
        __syncthreads();

        // Compute per-thread kTM x kTN tile
        #pragma unroll
        for (int k = 0; k < kBK; ++k) {
            float areg[kTM];
            float breg[kTN];
            #pragma unroll
            for (int i = 0; i < kTM; ++i) areg[i] = A[thread_row * kTM + i][k];
            #pragma unroll
            for (int j = 0; j < kTN; ++j) breg[j] = B[thread_col * kTN + j][k];
            #pragma unroll
            for (int i = 0; i < kTM; ++i)
                #pragma unroll
                for (int j = 0; j < kTN; ++j) acc[i][j] += areg[i] * breg[j];
        }
        __syncthreads();
    }

    // Write output with optional bias
    #pragma unroll
    for (int i = 0; i < kTM; ++i) {
        const int m = m_tile + thread_row * kTM + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < kTN; ++j) {
            const int n = n_tile + thread_col * kTN + j;
            if (n >= N) continue;
            float v = acc[i][j];
            if (bias != nullptr) v += static_cast<float>(bias[n]);
            Y[m * N + n] = static_cast<scalar_t>(v);
        }
    }
}

template <typename scalar_t>
__global__ void bias_grad_kernel(
    const scalar_t* __restrict__ grad_out,  // (M, N)
    scalar_t* __restrict__ grad_bias,        // (N,)
    int64_t M, int64_t N) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float s = 0.0f;
    for (int64_t i = 0; i < M; ++i) s += static_cast<float>(grad_out[i * N + n]);
    grad_bias[n] = static_cast<scalar_t>(s);
}
}  // namespace

torch::Tensor matmul_forward_cuda(
    const torch::Tensor& x, const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
    c10::cuda::CUDAGuard guard(x.device());

    const auto x_shape = x.sizes().vec();
    const int64_t K = x_shape.back();
    const int64_t N = weight.size(0);
    TORCH_CHECK(weight.size(1) == K, "weight inner dim must match x");

    auto x2d = x.reshape({-1, K});
    const int64_t M = x2d.size(0);
    auto out = torch::empty({M, N}, x.options());

    if (M == 0 || N == 0) {
        auto out_shape = x_shape;
        out_shape.back() = N;
        return out.reshape(out_shape);
    }

    const dim3 grid(static_cast<unsigned>(CEIL_DIV(N, kBN)), static_cast<unsigned>(CEIL_DIV(M, kBM)));
    const dim3 block(kThreadsPerBlock);
    const auto stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor bias_tensor;
    const bool has_bias = bias.has_value();
    if (has_bias) bias_tensor = bias->contiguous();

    DISPATCH_FLOAT_TYPES(x.scalar_type(), "matmul_fwd", [&] {
        const scalar_t* bias_ptr = has_bias ? bias_tensor.data_ptr<scalar_t>() : nullptr;
        matmul_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            x2d.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
            bias_ptr, out.data_ptr<scalar_t>(), M, N, K);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto out_shape = x_shape;
    out_shape.back() = N;
    return out.reshape(out_shape);
}

std::vector<torch::Tensor> matmul_backward_cuda(
    const torch::Tensor& x, const torch::Tensor& weight,
    const torch::Tensor& grad_out, bool needs_bias_grad) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    CHECK_INPUT(grad_out);
    c10::cuda::CUDAGuard guard(x.device());

    const auto x_shape = x.sizes().vec();
    const int64_t K = x_shape.back();
    const int64_t N = weight.size(0);
    auto x2d = x.reshape({-1, K});
    auto go2d = grad_out.reshape({-1, N});
    const int64_t M = x2d.size(0);

    // grad_x = grad_out @ weight: (M, N) @ (N, K) = (M, K)
    // Reuse matmul kernel by treating weight as (K, N) and grad_out as (M, N)
    // Actually grad_out @ weight means sum_n grad_out[m,n] * weight[n,k]
    // Our kernel does Y = X @ W^T: sum_k X[m,k] * W[n,k]. So to compute grad_x, we need
    // X = grad_out (M, N), W = weight.T (K, N). weight.T has shape (K, N), stored as weight.transpose.
    // Using our kernel: X=(M,N), W=(K,N) -> Y=(M,K). That matches!
    auto grad_x = torch::empty({M, K}, x.options());

    // We need weight transposed logically: shape (K, N), each row k contains weight[:, k].
    // weight is (N, K); weight.transpose(0, 1) is (K, N). Make contiguous.
    auto w_t = weight.transpose(0, 1).contiguous();  // (K, N)

    // Now launch matmul_fwd with X=go2d, W=w_t (i.e. matmul expects W shape (N_out, K_in)).
    // We want Y=(M, K). So our kernel's N_out = K_in_bwd = K, K_in = N.
    {
        const dim3 grid(static_cast<unsigned>(CEIL_DIV(K, kBN)), static_cast<unsigned>(CEIL_DIV(M, kBM)));
        const dim3 block(kThreadsPerBlock);
        const auto stream = at::cuda::getCurrentCUDAStream();
        DISPATCH_FLOAT_TYPES(x.scalar_type(), "matmul_bwd_gx", [&] {
            matmul_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
                go2d.data_ptr<scalar_t>(), w_t.data_ptr<scalar_t>(),
                nullptr, grad_x.data_ptr<scalar_t>(), M, K, N);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // grad_weight = grad_out^T @ x: (N, M) @ (M, K) = (N, K)
    // Our kernel computes X @ W^T. With X=grad_out^T (N, M), W=x (M, K) reshaped to (K, M)?
    // Easier: grad_weight[n, k] = sum_m grad_out[m, n] * x[m, k].
    // This is X=(N, M) @ W=(K, M) -> kernel does Y[n, k] = sum_m X[n, m] * W[k, m].
    // So X = go2d.transpose(0, 1) = (N, M), W = x2d.transpose(0, 1) = (K, M).
    // That computes: Y[n, k] = sum_m go[n, m] * x[k, m]. Wait, we need sum_m go[m, n] * x[m, k] = sum_m go_T[n, m] * x[m, k].
    // Let's re-derive: kernel does Y = X @ W^T. So Y[i, j] = sum_k X[i, k] * W[j, k].
    // We want grad_weight[n, k] = sum_m grad_out[m, n] * x[m, k].
    // Let i=n, j=k, sum_variable=m, X[n, m] = grad_out[m, n] (so X is go.T, shape (N, M)), W[k, m] = x[m, k] (so W is x.T, shape (K, M)).
    // kernel M_dim = N, N_dim = K, K_dim = M.
    auto grad_weight = torch::empty({N, K}, weight.options());
    auto go_t = go2d.transpose(0, 1).contiguous();  // (N, M)
    auto x_t = x2d.transpose(0, 1).contiguous();    // (K, M)
    {
        const dim3 grid(static_cast<unsigned>(CEIL_DIV(K, kBN)), static_cast<unsigned>(CEIL_DIV(N, kBM)));
        const dim3 block(kThreadsPerBlock);
        const auto stream = at::cuda::getCurrentCUDAStream();
        DISPATCH_FLOAT_TYPES(x.scalar_type(), "matmul_bwd_gw", [&] {
            matmul_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
                go_t.data_ptr<scalar_t>(), x_t.data_ptr<scalar_t>(),
                nullptr, grad_weight.data_ptr<scalar_t>(), N, K, M);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    torch::Tensor grad_bias;
    if (needs_bias_grad) {
        grad_bias = torch::empty({N}, weight.options());
        const int threads = 256;
        const int blocks = static_cast<int>(CEIL_DIV(N, threads));
        const auto stream = at::cuda::getCurrentCUDAStream();
        DISPATCH_FLOAT_TYPES(x.scalar_type(), "bias_grad", [&] {
            bias_grad_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                go2d.data_ptr<scalar_t>(), grad_bias.data_ptr<scalar_t>(), M, N);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        grad_bias = torch::Tensor();
    }

    return {grad_x.reshape(x_shape), grad_weight, grad_bias};
}
