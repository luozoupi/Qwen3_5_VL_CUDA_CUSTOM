#include "common.h"
#include <cfloat>

// MoE routing kernels:
//   1. topk: per-row top-k selection (specialized k<=8)
//   2. index_add: scatter-add by row index with atomic ops
//   3. batched_gemm: Y[e] = X[e] @ W[e].T for stacked experts

namespace {
// Top-k via repeated max (works for small k). Modifies a copy of input in-place.
template <typename scalar_t>
__global__ void topk_kernel(
    const scalar_t* __restrict__ x,   // (N, C) row-major
    scalar_t* __restrict__ work,       // (N, C) scratch, filled with x
    scalar_t* __restrict__ vals,       // (N, K)
    int64_t* __restrict__ idxs,        // (N, K)
    int64_t N, int64_t C, int64_t K) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const int blk = blockDim.x;
    scalar_t* w_row = work + row * C;
    // Copy to work buffer
    for (int i = tid; i < C; i += blk) w_row[i] = x[row * C + i];
    __syncthreads();

    __shared__ float s_vals[32];
    __shared__ int s_idxs[32];

    for (int64_t ki = 0; ki < K; ++ki) {
        // Find max of w_row
        float best_v = -FLT_MAX;
        int best_i = 0;
        for (int i = tid; i < C; i += blk) {
            const float v = static_cast<float>(w_row[i]);
            if (v > best_v) { best_v = v; best_i = i; }
        }
        // Warp reduce
        for (int off = 16; off > 0; off /= 2) {
            float o_v = __shfl_down_sync(0xffffffff, best_v, off);
            int o_i = __shfl_down_sync(0xffffffff, best_i, off);
            if (o_v > best_v) { best_v = o_v; best_i = o_i; }
        }
        const int lane = tid % 32, warp = tid / 32;
        if (lane == 0) { s_vals[warp] = best_v; s_idxs[warp] = best_i; }
        __syncthreads();
        if (warp == 0) {
            best_v = (tid < (blk + 31) / 32) ? s_vals[lane] : -FLT_MAX;
            best_i = (tid < (blk + 31) / 32) ? s_idxs[lane] : 0;
            for (int off = 16; off > 0; off /= 2) {
                float o_v = __shfl_down_sync(0xffffffff, best_v, off);
                int o_i = __shfl_down_sync(0xffffffff, best_i, off);
                if (o_v > best_v) { best_v = o_v; best_i = o_i; }
            }
            if (lane == 0) {
                vals[row * K + ki] = static_cast<scalar_t>(best_v);
                idxs[row * K + ki] = static_cast<int64_t>(best_i);
                w_row[best_i] = static_cast<scalar_t>(-FLT_MAX);  // exclude from next iter
            }
        }
        __syncthreads();
    }
}

// Proper atomic add for float32 / half / bfloat16 / double
template <typename scalar_t>
__device__ __forceinline__ void atomic_add_scalar(scalar_t* addr, float val);

template <>
__device__ __forceinline__ void atomic_add_scalar<float>(float* addr, float val) {
    atomicAdd(addr, val);
}

template <>
__device__ __forceinline__ void atomic_add_scalar<double>(double* addr, float val) {
    atomicAdd(addr, static_cast<double>(val));
}

template <>
__device__ __forceinline__ void atomic_add_scalar<at::Half>(at::Half* addr, float val) {
    atomicAdd(reinterpret_cast<__half*>(addr), __float2half(val));
}

template <>
__device__ __forceinline__ void atomic_add_scalar<at::BFloat16>(at::BFloat16* addr, float val) {
    atomicAdd(reinterpret_cast<__nv_bfloat16*>(addr), __float2bfloat16(val));
}

template <typename scalar_t>
__global__ void index_add_safe_kernel(
    scalar_t* __restrict__ target,
    const scalar_t* __restrict__ source,
    const int64_t* __restrict__ index,
    int64_t S, int64_t D) {
    const int src_row = blockIdx.x;
    if (src_row >= S) return;
    const int64_t tgt_row = index[src_row];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        atomic_add_scalar<scalar_t>(&target[tgt_row * D + d],
                                    static_cast<float>(source[src_row * D + d]));
    }
}

// Batched GEMM: Y[e, m, n] = sum_k X[e, m, k] * W[e, n, k]  (W is (E, N, K))
template <typename scalar_t>
__global__ void batched_gemm_kernel(
    const scalar_t* __restrict__ X,   // (E, M, K)
    const scalar_t* __restrict__ W,   // (E, N, K)
    scalar_t* __restrict__ Y,          // (E, M, N)
    int64_t E, int64_t M, int64_t N, int64_t K) {
    const int e = blockIdx.z;
    const int m_block = blockIdx.y;
    const int n_block = blockIdx.x;
    const int tid = threadIdx.x;

    const int M_TILE = 16, N_TILE = 16, K_TILE = 16;
    __shared__ float A[M_TILE][K_TILE];
    __shared__ float B[N_TILE][K_TILE];

    const int m_start = m_block * M_TILE;
    const int n_start = n_block * N_TILE;

    const int row_in_tile = tid / N_TILE;
    const int col_in_tile = tid % N_TILE;

    if (row_in_tile >= M_TILE) return;

    float acc = 0.0f;
    for (int k_start = 0; k_start < K; k_start += K_TILE) {
        // Cooperative load
        for (int i = tid; i < M_TILE * K_TILE; i += blockDim.x) {
            const int r = i / K_TILE;
            const int c = i % K_TILE;
            const int64_t gm = m_start + r;
            const int64_t gk = k_start + c;
            A[r][c] = (gm < M && gk < K) ? static_cast<float>(X[((int64_t)e * M + gm) * K + gk]) : 0.0f;
        }
        for (int i = tid; i < N_TILE * K_TILE; i += blockDim.x) {
            const int r = i / K_TILE;
            const int c = i % K_TILE;
            const int64_t gn = n_start + r;
            const int64_t gk = k_start + c;
            B[r][c] = (gn < N && gk < K) ? static_cast<float>(W[((int64_t)e * N + gn) * K + gk]) : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < K_TILE; ++k) acc += A[row_in_tile][k] * B[col_in_tile][k];
        __syncthreads();
    }

    const int64_t gm = m_start + row_in_tile;
    const int64_t gn = n_start + col_in_tile;
    if (gm < M && gn < N) {
        Y[((int64_t)e * M + gm) * N + gn] = static_cast<scalar_t>(acc);
    }
}
}  // namespace

std::vector<torch::Tensor> topk_forward_cuda(const torch::Tensor& x, int64_t k) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "topk expects 2D input");
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    auto vals = torch::empty({N, k}, x.options());
    auto idxs = torch::empty({N, k}, x.options().dtype(torch::kInt64));
    auto work = torch::empty_like(x);
    if (N == 0) return {vals, idxs};

    const int threads = std::min<int64_t>(1024, ((C + 31) / 32) * 32);
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(x.scalar_type(), "topk", [&] {
        topk_kernel<scalar_t><<<static_cast<int>(N), threads, 0, stream>>>(
            x.data_ptr<scalar_t>(), work.data_ptr<scalar_t>(),
            vals.data_ptr<scalar_t>(), idxs.data_ptr<int64_t>(),
            N, C, k);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {vals, idxs};
}

torch::Tensor index_add_forward_cuda(
    torch::Tensor target, const torch::Tensor& source, const torch::Tensor& index) {
    CHECK_INPUT(target);
    CHECK_INPUT(source);
    CHECK_INPUT(index);
    TORCH_CHECK(target.dim() == 2 && source.dim() == 2, "target, source must be 2D");
    c10::cuda::CUDAGuard guard(target.device());
    const int64_t S = source.size(0);
    const int64_t D = source.size(1);
    if (S == 0) return target;
    const int threads = std::min<int64_t>(256, D);
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(target.scalar_type(), "index_add", [&] {
        index_add_safe_kernel<scalar_t><<<static_cast<int>(S), threads, 0, stream>>>(
            target.data_ptr<scalar_t>(), source.data_ptr<scalar_t>(),
            index.data_ptr<int64_t>(), S, D);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return target;
}

torch::Tensor batched_gemm_forward_cuda(const torch::Tensor& x, const torch::Tensor& weight) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    TORCH_CHECK(x.dim() == 3, "x must be (E, M, K)");
    TORCH_CHECK(weight.dim() == 3, "weight must be (E, N, K)");
    TORCH_CHECK(x.size(0) == weight.size(0), "expert count must match");
    TORCH_CHECK(x.size(2) == weight.size(2), "inner dim must match");
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t E = x.size(0);
    const int64_t M = x.size(1);
    const int64_t K = x.size(2);
    const int64_t N = weight.size(1);
    auto out = torch::empty({E, M, N}, x.options());
    if (E * M * N == 0) return out;

    const int M_TILE = 16, N_TILE = 16;
    const dim3 grid((unsigned)CEIL_DIV(N, N_TILE), (unsigned)CEIL_DIV(M, M_TILE), (unsigned)E);
    const dim3 block(M_TILE * N_TILE);
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(x.scalar_type(), "batched_gemm", [&] {
        batched_gemm_kernel<scalar_t><<<grid, block, 0, stream>>>(
            x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), E, M, N, K);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
