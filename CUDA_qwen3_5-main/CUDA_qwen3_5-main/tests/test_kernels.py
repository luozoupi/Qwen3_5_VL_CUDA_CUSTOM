"""Per-op parity tests for every CUDA kernel vs PyTorch reference."""
import pytest
import torch
import torch.nn.functional as F

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# -------------------- Elementwise --------------------

@cuda_only
def test_residual_add():
    from cuda_qwen3_vl.kernels import residual_add
    a = torch.randn(4, 16, 64, device="cuda", dtype=torch.float32)
    b = torch.randn(4, 16, 64, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(residual_add(a, b), a + b, atol=1e-5, rtol=1e-5)


@cuda_only
def test_sigmoid_mul():
    from cuda_qwen3_vl.kernels import sigmoid_mul
    x = torch.randn(8, 128, device="cuda", dtype=torch.float32)
    gate = torch.randn(8, 128, device="cuda", dtype=torch.float32)
    expected = x * torch.sigmoid(gate)
    torch.testing.assert_close(sigmoid_mul(x, gate), expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_sigmoid_mul_backward():
    from cuda_qwen3_vl.kernels import sigmoid_mul
    x = torch.randn(4, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    g = torch.randn(4, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    rx, rg = x.detach().clone().requires_grad_(True), g.detach().clone().requires_grad_(True)
    grad = torch.randn(4, 64, device="cuda")
    sigmoid_mul(x, g).backward(grad)
    (rx * torch.sigmoid(rg)).backward(grad)
    torch.testing.assert_close(x.grad, rx.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(g.grad, rg.grad, atol=1e-4, rtol=1e-4)


@cuda_only
def test_gelu_tanh():
    from cuda_qwen3_vl.kernels import gelu_tanh
    x = torch.randn(8, 128, device="cuda", dtype=torch.float32)
    expected = F.gelu(x, approximate="tanh")
    torch.testing.assert_close(gelu_tanh(x), expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_swiglu():
    from cuda_qwen3_vl.kernels import swiglu
    gate = torch.randn(8, 128, device="cuda", dtype=torch.float32)
    up = torch.randn(8, 128, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(swiglu(gate, up), F.silu(gate) * up, atol=1e-4, rtol=1e-4)


# -------------------- Norms --------------------

@cuda_only
def test_rmsnorm():
    from cuda_qwen3_vl.kernels import rmsnorm
    x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
    w = torch.randn(256, device="cuda", dtype=torch.float32)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(rmsnorm(x, w, 1e-6), expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_rmsnorm_backward():
    from cuda_qwen3_vl.kernels import rmsnorm
    x = torch.randn(4, 256, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(256, device="cuda", dtype=torch.float32, requires_grad=True)
    rx, rw = x.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True)
    grad = torch.randn(4, 256, device="cuda")
    rmsnorm(x, w, 1e-6).backward(grad)
    (rx * torch.rsqrt(rx.pow(2).mean(-1, keepdim=True) + 1e-6) * rw).backward(grad)
    torch.testing.assert_close(x.grad, rx.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(w.grad, rw.grad, atol=1e-4, rtol=1e-4)


@cuda_only
def test_layernorm():
    from cuda_qwen3_vl.kernels import layernorm
    x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
    w = torch.randn(256, device="cuda", dtype=torch.float32)
    b = torch.randn(256, device="cuda", dtype=torch.float32)
    expected = F.layer_norm(x, (256,), w, b, 1e-6)
    torch.testing.assert_close(layernorm(x, w, b, 1e-6), expected, atol=1e-4, rtol=1e-4)


# -------------------- Embedding / Reduction --------------------

@cuda_only
def test_embedding():
    from cuda_qwen3_vl.kernels import embedding
    w = torch.randn(100, 64, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 100, (4, 16), device="cuda", dtype=torch.int64)
    expected = F.embedding(ids, w)
    torch.testing.assert_close(embedding(ids, w), expected, atol=1e-5, rtol=1e-5)


@cuda_only
def test_softmax():
    from cuda_qwen3_vl.kernels import softmax
    x = torch.randn(8, 256, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-5)


@cuda_only
def test_cross_entropy():
    from cuda_qwen3_vl.kernels import cross_entropy
    logits = torch.randn(16, 100, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, 100, (16,), device="cuda", dtype=torch.int64)
    expected = F.cross_entropy(logits, targets)
    torch.testing.assert_close(cross_entropy(logits, targets), expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_cross_entropy_backward():
    from cuda_qwen3_vl.kernels import cross_entropy
    logits = torch.randn(16, 100, device="cuda", dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, 100, (16,), device="cuda", dtype=torch.int64)
    rlogits = logits.detach().clone().requires_grad_(True)
    cross_entropy(logits, targets).backward()
    F.cross_entropy(rlogits, targets).backward()
    torch.testing.assert_close(logits.grad, rlogits.grad, atol=1e-4, rtol=1e-4)


# -------------------- RoPE --------------------

def _torch_rope(x, cos, sin):
    D_rope = cos.shape[-1]
    half = D_rope // 2
    x_rot = x[..., :D_rope]
    x_pass = x[..., D_rope:]
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    c = cos.unsqueeze(1) if cos.dim() == 3 else cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(1) if sin.dim() == 3 else sin.unsqueeze(0).unsqueeze(0)
    c1, s1 = c[..., :half], s[..., :half]
    rot = torch.cat([x1 * c1 - x2 * s1, x2 * c1 + x1 * s1], dim=-1)
    return torch.cat([rot, x_pass], dim=-1)


@cuda_only
def test_rope():
    from cuda_qwen3_vl.kernels import apply_rope
    x = torch.randn(2, 4, 8, 32, device="cuda", dtype=torch.float32)
    cos = torch.randn(2, 8, 32, device="cuda", dtype=torch.float32)
    sin = torch.randn(2, 8, 32, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(apply_rope(x, cos, sin), _torch_rope(x, cos, sin), atol=1e-4, rtol=1e-4)


# -------------------- GEMM --------------------

@cuda_only
def test_matmul():
    from cuda_qwen3_vl.kernels import matmul
    x = torch.randn(8, 64, device="cuda", dtype=torch.float32)
    w = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    b = torch.randn(32, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(matmul(x, w, b), F.linear(x, w, b), atol=1e-3, rtol=1e-3)


@cuda_only
def test_matmul_backward():
    from cuda_qwen3_vl.kernels import matmul
    x = torch.randn(8, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(32, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn(32, device="cuda", dtype=torch.float32, requires_grad=True)
    rx, rw, rb = (x.detach().clone().requires_grad_(True),
                  w.detach().clone().requires_grad_(True),
                  b.detach().clone().requires_grad_(True))
    grad = torch.randn(8, 32, device="cuda")
    matmul(x, w, b).backward(grad)
    F.linear(rx, rw, rb).backward(grad)
    torch.testing.assert_close(x.grad, rx.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(w.grad, rw.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(b.grad, rb.grad, atol=1e-3, rtol=1e-3)


# -------------------- Flash Attention --------------------

@cuda_only
def test_flash_attention_causal():
    from cuda_qwen3_vl.kernels import flash_attention
    torch.manual_seed(42)
    q = torch.randn(2, 4, 16, 32, device="cuda", dtype=torch.float32)
    k = torch.randn(2, 4, 16, 32, device="cuda", dtype=torch.float32)
    v = torch.randn(2, 4, 16, 32, device="cuda", dtype=torch.float32)
    scale = 32 ** -0.5
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    actual = flash_attention(q, k, v, scale=scale, is_causal=True, num_kv_groups=1)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@cuda_only
def test_flash_attention_non_causal():
    from cuda_qwen3_vl.kernels import flash_attention
    q = torch.randn(2, 4, 16, 32, device="cuda", dtype=torch.float32)
    k = torch.randn(2, 4, 16, 32, device="cuda", dtype=torch.float32)
    v = torch.randn(2, 4, 16, 32, device="cuda", dtype=torch.float32)
    scale = 32 ** -0.5
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)
    actual = flash_attention(q, k, v, scale=scale, is_causal=False, num_kv_groups=1)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@cuda_only
def test_flash_attention_backward_causal():
    from cuda_qwen3_vl.kernels import flash_attention
    torch.manual_seed(0)
    q = torch.randn(2, 4, 16, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(2, 4, 16, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    v = torch.randn(2, 4, 16, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    scale = 32 ** -0.5

    rq = q.detach().clone().requires_grad_(True)
    rk = k.detach().clone().requires_grad_(True)
    rv = v.detach().clone().requires_grad_(True)

    out = flash_attention(q, k, v, scale=scale, is_causal=True, num_kv_groups=1)
    ref = F.scaled_dot_product_attention(rq, rk, rv, is_causal=True, scale=scale)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)

    torch.testing.assert_close(q.grad, rq.grad, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(k.grad, rk.grad, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(v.grad, rv.grad, atol=5e-3, rtol=5e-3)


@cuda_only
def test_flash_attention_backward_gqa():
    from cuda_qwen3_vl.kernels import flash_attention
    torch.manual_seed(0)
    q = torch.randn(2, 8, 16, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(2, 2, 16, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    v = torch.randn(2, 2, 16, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    scale = 32 ** -0.5

    rq = q.detach().clone().requires_grad_(True)
    rk = k.detach().clone().requires_grad_(True)
    rv = v.detach().clone().requires_grad_(True)
    k_exp = rk[:, :, None].expand(2, 2, 4, 16, 32).reshape(2, 8, 16, 32)
    v_exp = rv[:, :, None].expand(2, 2, 4, 16, 32).reshape(2, 8, 16, 32)

    out = flash_attention(q, k, v, scale=scale, is_causal=True, num_kv_groups=4)
    ref = F.scaled_dot_product_attention(rq, k_exp, v_exp, is_causal=True, scale=scale)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)

    torch.testing.assert_close(q.grad, rq.grad, atol=5e-3, rtol=5e-3)
    # dK/dV: the GQA reference backward propagates via expand — grads are summed
    # across the 4 replicated heads. Our kernel sums all 4 Q-heads into one KV-head.
    torch.testing.assert_close(k.grad, rk.grad, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(v.grad, rv.grad, atol=5e-3, rtol=5e-3)


@cuda_only
def test_flash_attention_gqa():
    from cuda_qwen3_vl.kernels import flash_attention
    q = torch.randn(2, 8, 16, 32, device="cuda", dtype=torch.float32)
    k = torch.randn(2, 2, 16, 32, device="cuda", dtype=torch.float32)
    v = torch.randn(2, 2, 16, 32, device="cuda", dtype=torch.float32)
    scale = 32 ** -0.5
    # Expand for reference
    k_exp = k[:, :, None].expand(2, 2, 4, 16, 32).reshape(2, 8, 16, 32)
    v_exp = v[:, :, None].expand(2, 2, 4, 16, 32).reshape(2, 8, 16, 32)
    expected = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True, scale=scale)
    actual = flash_attention(q, k, v, scale=scale, is_causal=True, num_kv_groups=4)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


# -------------------- MoE routing --------------------

@cuda_only
def test_topk():
    from cuda_qwen3_vl.kernels import cuda_topk
    x = torch.randn(8, 60, device="cuda", dtype=torch.float32)
    v, i = cuda_topk(x, 4)
    rv, ri = torch.topk(x, 4, dim=-1)
    torch.testing.assert_close(v, rv, atol=1e-5, rtol=1e-5)
    assert (i == ri).all()


@cuda_only
def test_index_add():
    from cuda_qwen3_vl.kernels import cuda_index_add
    tgt = torch.zeros(8, 32, device="cuda", dtype=torch.float32)
    src = torch.randn(5, 32, device="cuda", dtype=torch.float32)
    idx = torch.tensor([0, 2, 4, 6, 7], device="cuda", dtype=torch.int64)
    rtgt = tgt.clone()
    rtgt.index_add_(0, idx, src)
    cuda_index_add(tgt, src, idx)
    torch.testing.assert_close(tgt, rtgt, atol=1e-5, rtol=1e-5)


@cuda_only
def test_batched_gemm():
    from cuda_qwen3_vl.kernels import cuda_batched_gemm
    x = torch.randn(4, 8, 16, device="cuda", dtype=torch.float32)  # (E, M, K)
    w = torch.randn(4, 12, 16, device="cuda", dtype=torch.float32)  # (E, N, K)
    expected = torch.einsum("emk,enk->emn", x, w)
    actual = cuda_batched_gemm(x, w)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


# -------------------- Conv3d patch --------------------

@cuda_only
def test_conv3d_patch():
    from cuda_qwen3_vl.kernels import conv3d_patch
    N, C, T, H, W = 4, 3, 2, 16, 16
    E = 32
    x = torch.randn(N, C, T, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(E, C, T, H, W, device="cuda", dtype=torch.float32)
    bias = torch.randn(E, device="cuda", dtype=torch.float32)
    # Reference via F.conv3d with stride=kernel_size
    expected = F.conv3d(x, weight, bias=bias, stride=(T, H, W)).reshape(N, E)
    actual = conv3d_patch(x, weight, bias)
    # Wide tolerance: dot product over K=1536 accumulates fp32 rounding differently
    # between our simple row-dot and cuDNN's conv3d.
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
