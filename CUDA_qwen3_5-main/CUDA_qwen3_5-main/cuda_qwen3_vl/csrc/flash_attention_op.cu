#include "common.h"
#include <cfloat>

// Flash Attention v2 forward (tiled, online softmax).
// Inputs:
//   Q: (B, H_q, S_q, D)
//   K: (B, H_kv, S_k, D)  — GQA: H_q = H_kv * num_kv_groups
//   V: (B, H_kv, S_k, D)
// Output: (B, H_q, S_q, D)
// scale: softmax scale (1/sqrt(D) typically)
// is_causal: causal mask inside kernel (col <= row)

namespace {
constexpr int kBlockM = 16;
constexpr int kBlockN = 32;

template <typename scalar_t, int BLOCK_D>
__global__ void flash_attn_fwd_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ O,
    float* __restrict__ LSE,  // (B, H_q, S_q) — logsumexp for backward
    int64_t B, int64_t H_q, int64_t H_kv, int64_t S_q, int64_t S_k, int64_t D,
    int64_t num_kv_groups, float scale, bool is_causal) {
    const int pid_m = blockIdx.x;  // Q tile (rows of size kBlockM)
    const int pid_bh = blockIdx.y; // batch * H_q
    const int tid = threadIdx.x;

    const int64_t b = pid_bh / H_q;
    const int64_t hq = pid_bh % H_q;
    const int64_t hkv = hq / num_kv_groups;

    if (b >= B) return;

    const scalar_t* Q_bh = Q + ((b * H_q + hq) * S_q) * D;
    const scalar_t* K_bh = K + ((b * H_kv + hkv) * S_k) * D;
    const scalar_t* V_bh = V + ((b * H_kv + hkv) * S_k) * D;
    scalar_t* O_bh = O + ((b * H_q + hq) * S_q) * D;
    float* LSE_bh = LSE + (b * H_q + hq) * S_q;

    // Shared memory: Q tile (kBlockM x BLOCK_D), K tile (kBlockN x BLOCK_D), V tile (kBlockN x BLOCK_D)
    __shared__ float Qs[kBlockM][BLOCK_D];
    __shared__ float Ks[kBlockN][BLOCK_D];
    __shared__ float Vs[kBlockN][BLOCK_D];
    __shared__ float Ss[kBlockM][kBlockN];  // attention scores

    // Each thread handles one Q row within the Q block
    const int qrow = tid;  // 0..kBlockM-1 (assume blockDim.x == kBlockM)
    const int64_t global_q = pid_m * kBlockM + qrow;
    const bool valid_q = global_q < S_q;

    // Load Q tile
    if (valid_q) {
        for (int d = 0; d < D && d < BLOCK_D; ++d) {
            Qs[qrow][d] = static_cast<float>(Q_bh[global_q * D + d]);
        }
        for (int d = D; d < BLOCK_D; ++d) Qs[qrow][d] = 0.0f;
    } else {
        for (int d = 0; d < BLOCK_D; ++d) Qs[qrow][d] = 0.0f;
    }

    // Running softmax state
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float acc[BLOCK_D];
    #pragma unroll
    for (int d = 0; d < BLOCK_D; ++d) acc[d] = 0.0f;

    const int64_t kv_end = is_causal
        ? min((int64_t)((pid_m + 1) * kBlockM), S_k)
        : S_k;

    for (int64_t kv_start = 0; kv_start < kv_end; kv_start += kBlockN) {
        __syncthreads();
        // Load K, V tiles cooperatively
        const int items = kBlockN * BLOCK_D;
        for (int i = tid; i < items; i += blockDim.x) {
            const int n = i / BLOCK_D;
            const int d = i % BLOCK_D;
            const int64_t global_n = kv_start + n;
            if (global_n < S_k && d < D) {
                Ks[n][d] = static_cast<float>(K_bh[global_n * D + d]);
                Vs[n][d] = static_cast<float>(V_bh[global_n * D + d]);
            } else {
                Ks[n][d] = 0.0f;
                Vs[n][d] = 0.0f;
            }
        }
        __syncthreads();

        if (!valid_q) continue;

        // Compute scores S[qrow, n] = dot(Q[qrow], K[n]) * scale
        for (int n = 0; n < kBlockN; ++n) {
            float s = 0.0f;
            for (int d = 0; d < D && d < BLOCK_D; ++d) s += Qs[qrow][d] * Ks[n][d];
            s *= scale;
            const int64_t global_n = kv_start + n;
            if (global_n >= S_k) s = -FLT_MAX;
            if (is_causal && global_n > global_q) s = -FLT_MAX;
            Ss[qrow][n] = s;
        }

        // Online softmax update
        float m_ij = -FLT_MAX;
        for (int n = 0; n < kBlockN; ++n) m_ij = fmaxf(m_ij, Ss[qrow][n]);
        const float m_new = fmaxf(m_i, m_ij);
        const float alpha = (m_i == -FLT_MAX) ? 0.0f : __expf(m_i - m_new);
        float l_ij = 0.0f;
        for (int n = 0; n < kBlockN; ++n) {
            Ss[qrow][n] = __expf(Ss[qrow][n] - m_new);
            l_ij += Ss[qrow][n];
        }
        l_i = l_i * alpha + l_ij;
        for (int d = 0; d < D && d < BLOCK_D; ++d) acc[d] *= alpha;
        // acc += P @ V for this Q row
        for (int n = 0; n < kBlockN; ++n) {
            const float p = Ss[qrow][n];
            for (int d = 0; d < D && d < BLOCK_D; ++d) {
                acc[d] += p * Vs[n][d];
            }
        }
        m_i = m_new;
    }

    // Write output
    if (valid_q) {
        const float inv_l = 1.0f / l_i;
        for (int d = 0; d < D && d < BLOCK_D; ++d) {
            O_bh[global_q * D + d] = static_cast<scalar_t>(acc[d] * inv_l);
        }
        LSE_bh[global_q] = m_i + logf(l_i);
    }
}
}  // namespace

std::vector<torch::Tensor> flash_attention_forward_cuda(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    double scale, bool is_causal, int64_t num_kv_groups) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    c10::cuda::CUDAGuard guard(q.device());

    const int64_t B = q.size(0);
    const int64_t H_q = q.size(1);
    const int64_t H_kv = k.size(1);
    const int64_t S_q = q.size(2);
    const int64_t S_k = k.size(2);
    const int64_t D = q.size(3);
    TORCH_CHECK(D <= 128, "head dim > 128 not supported (got ", D, ")");

    auto out = torch::empty_like(q);
    auto lse = torch::empty({B, H_q, S_q}, q.options().dtype(torch::kFloat32));

    const dim3 grid(static_cast<unsigned>(CEIL_DIV(S_q, kBlockM)),
                    static_cast<unsigned>(B * H_q));
    const dim3 block(kBlockM);  // one thread per Q row
    const auto stream = at::cuda::getCurrentCUDAStream();

    auto launch = [&](auto block_d_tag) {
        constexpr int BD = decltype(block_d_tag)::value;
        DISPATCH_FLOAT_TYPES(q.scalar_type(), "flash_attn_fwd", [&] {
            flash_attn_fwd_kernel<scalar_t, BD><<<grid, block, 0, stream>>>(
                q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), lse.data_ptr<float>(),
                B, H_q, H_kv, S_q, S_k, D, num_kv_groups,
                static_cast<float>(scale), is_causal);
        });
    };

    if (D <= 32) launch(std::integral_constant<int, 32>{});
    else if (D <= 64) launch(std::integral_constant<int, 64>{});
    else if (D <= 96) launch(std::integral_constant<int, 96>{});
    else launch(std::integral_constant<int, 128>{});

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, lse};
}

// -----------------------------------------------------------------------------
// Flash Attention backward
//
// Given Q, K, V, O, LSE, dO — compute dQ, dK, dV.
//
// Strategy: one thread-block per (batch, q_head, q_position). Each block handles
// a single Q row end-to-end, walking the K/V sequence in tiles. Within a block,
// kNThreadsBwd threads cooperate to parallelize over the head_dim.
//
// Per Q-row work:
//   1) Recompute delta = sum_k O[q_row, k] * dO[q_row, k]
//   2) For each k_row in 0..S_k:
//        - Recompute S = (Q[q_row] . K[k_row]) * scale
//        - P = exp(S - LSE[q_row])
//        - dV[k_row] += P * dO[q_row]              (atomic)
//        - dP = dO[q_row] . V[k_row]
//        - dS = P * (dP - delta) * scale
//        - dQ[q_row] += dS * K[k_row]               (local accum)
//        - dK[k_row] += dS * Q[q_row]               (atomic)
//   3) Write dQ[q_row]
//
// This is O(S_q * S_k * D) compute — same cost class as the reference backward
// via SDPA + autograd — but fully on our CUDA kernel, no PyTorch fallback.
// -----------------------------------------------------------------------------

namespace {
constexpr int kNThreadsBwd = 128;

template <typename scalar_t>
__device__ __forceinline__ float atomic_add_scalar_bwd(scalar_t* addr, float val);

template <>
__device__ __forceinline__ float atomic_add_scalar_bwd<float>(float* addr, float val) {
    atomicAdd(addr, val);
    return 0.0f;
}
template <>
__device__ __forceinline__ float atomic_add_scalar_bwd<double>(double* addr, float val) {
    atomicAdd(addr, static_cast<double>(val));
    return 0.0f;
}
template <>
__device__ __forceinline__ float atomic_add_scalar_bwd<at::Half>(at::Half* addr, float val) {
    atomicAdd(reinterpret_cast<__half*>(addr), __float2half(val));
    return 0.0f;
}
template <>
__device__ __forceinline__ float atomic_add_scalar_bwd<at::BFloat16>(at::BFloat16* addr, float val) {
    atomicAdd(reinterpret_cast<__nv_bfloat16*>(addr), __float2bfloat16(val));
    return 0.0f;
}

template <typename scalar_t>
__global__ void flash_attn_bwd_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ O,
    const scalar_t* __restrict__ dO,
    const float* __restrict__ LSE,
    scalar_t* __restrict__ dQ,
    scalar_t* __restrict__ dK,
    scalar_t* __restrict__ dV,
    int64_t B, int64_t H_q, int64_t H_kv, int64_t S_q, int64_t S_k, int64_t D,
    int64_t num_kv_groups, float scale, bool is_causal) {
    const int pid_bh = blockIdx.y;
    const int pid_sq = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int64_t b = pid_bh / H_q;
    const int64_t hq = pid_bh % H_q;
    const int64_t hkv = hq / num_kv_groups;
    const int64_t sq = pid_sq;
    if (b >= B || sq >= S_q) return;

    // Pointers to this row in each tensor.
    const scalar_t* q_row = Q + ((b * H_q + hq) * S_q + sq) * D;
    const scalar_t* do_row = dO + ((b * H_q + hq) * S_q + sq) * D;
    const scalar_t* o_row = O + ((b * H_q + hq) * S_q + sq) * D;
    scalar_t* dq_row = dQ + ((b * H_q + hq) * S_q + sq) * D;

    // Shared memory: Q row + dO row + delta scalar + dq row accumulator (fp32).
    extern __shared__ float smem[];
    float* q_fp = smem;                       // D
    float* do_fp = smem + D;                  // D
    float* dq_fp = smem + 2 * D;              // D
    float* reduce = smem + 3 * D;             // nthreads

    // Load Q row + dO row; initialize dq accum + compute delta.
    float thread_delta = 0.0f;
    for (int d = tid; d < D; d += nthreads) {
        float qv = static_cast<float>(q_row[d]);
        float dov = static_cast<float>(do_row[d]);
        float ov = static_cast<float>(o_row[d]);
        q_fp[d] = qv;
        do_fp[d] = dov;
        dq_fp[d] = 0.0f;
        thread_delta += ov * dov;
    }
    // Block reduce delta.
    reduce[tid] = thread_delta;
    __syncthreads();
    for (int off = nthreads / 2; off > 0; off >>= 1) {
        if (tid < off) reduce[tid] += reduce[tid + off];
        __syncthreads();
    }
    const float delta = reduce[0];
    __syncthreads();

    const float lse = LSE[(b * H_q + hq) * S_q + sq];

    const int64_t sk_end = is_causal ? (sq + 1) : S_k;

    for (int64_t sk = 0; sk < sk_end; ++sk) {
        const scalar_t* k_row = K + ((b * H_kv + hkv) * S_k + sk) * D;
        const scalar_t* v_row = V + ((b * H_kv + hkv) * S_k + sk) * D;
        scalar_t* dk_row = dK + ((b * H_kv + hkv) * S_k + sk) * D;
        scalar_t* dv_row = dV + ((b * H_kv + hkv) * S_k + sk) * D;

        // Compute S = Q . K, and dP = dO . V, with one pass.
        float thread_s = 0.0f;
        float thread_dp = 0.0f;
        for (int d = tid; d < D; d += nthreads) {
            float kv = static_cast<float>(k_row[d]);
            float vv = static_cast<float>(v_row[d]);
            thread_s += q_fp[d] * kv;
            thread_dp += do_fp[d] * vv;
        }
        reduce[tid] = thread_s;
        __syncthreads();
        for (int off = nthreads / 2; off > 0; off >>= 1) {
            if (tid < off) reduce[tid] += reduce[tid + off];
            __syncthreads();
        }
        const float S = reduce[0] * scale;
        __syncthreads();

        reduce[tid] = thread_dp;
        __syncthreads();
        for (int off = nthreads / 2; off > 0; off >>= 1) {
            if (tid < off) reduce[tid] += reduce[tid + off];
            __syncthreads();
        }
        const float dP = reduce[0];
        __syncthreads();

        const float P = __expf(S - lse);
        const float dS = P * (dP - delta) * scale;

        for (int d = tid; d < D; d += nthreads) {
            // dV[sk, d] += P * dO[sq, d]
            atomic_add_scalar_bwd<scalar_t>(&dv_row[d], P * do_fp[d]);
            // dQ[sq, d] += dS * K[sk, d] (local)
            dq_fp[d] += dS * static_cast<float>(k_row[d]);
            // dK[sk, d] += dS * Q[sq, d]
            atomic_add_scalar_bwd<scalar_t>(&dk_row[d], dS * q_fp[d]);
        }
        __syncthreads();
    }

    // Write dQ row.
    for (int d = tid; d < D; d += nthreads) {
        dq_row[d] = static_cast<scalar_t>(dq_fp[d]);
    }
}
}  // namespace

std::vector<torch::Tensor> flash_attention_backward_cuda(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& o, const torch::Tensor& lse, const torch::Tensor& grad_o,
    double scale, bool is_causal, int64_t num_kv_groups) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);
    CHECK_INPUT(grad_o);
    c10::cuda::CUDAGuard guard(q.device());

    const int64_t B = q.size(0);
    const int64_t H_q = q.size(1);
    const int64_t H_kv = k.size(1);
    const int64_t S_q = q.size(2);
    const int64_t S_k = k.size(2);
    const int64_t D = q.size(3);
    TORCH_CHECK(D <= 128, "head dim > 128 not supported");
    TORCH_CHECK(num_kv_groups * H_kv == H_q, "H_q must equal num_kv_groups * H_kv");

    auto dq = torch::zeros_like(q);
    auto dk = torch::zeros_like(k);
    auto dv = torch::zeros_like(v);

    const dim3 grid(static_cast<unsigned>(S_q), static_cast<unsigned>(B * H_q));
    const dim3 block(kNThreadsBwd);
    // Shared memory: q_fp + do_fp + dq_fp + reduce = (3*D + nthreads) floats
    const size_t shmem_bytes = (3 * D + kNThreadsBwd) * sizeof(float);
    const auto stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_FLOAT_TYPES(q.scalar_type(), "flash_attn_bwd", [&] {
        flash_attn_bwd_kernel<scalar_t><<<grid, block, shmem_bytes, stream>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            o.data_ptr<scalar_t>(), grad_o.data_ptr<scalar_t>(), lse.data_ptr<float>(),
            dq.data_ptr<scalar_t>(), dk.data_ptr<scalar_t>(), dv.data_ptr<scalar_t>(),
            B, H_q, H_kv, S_q, S_k, D, num_kv_groups,
            static_cast<float>(scale), is_causal);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dq, dk, dv};
}
