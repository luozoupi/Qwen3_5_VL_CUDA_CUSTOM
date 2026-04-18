"""Module-level sanity tests: CudaLinear, CudaEmbedding, norms, MLP, attention."""
import pytest
import torch
import torch.nn.functional as F

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda_only
def test_cuda_linear_matches_nn_linear():
    from cuda_qwen3_vl.modules import CudaLinear
    torch.manual_seed(0)
    ref = torch.nn.Linear(64, 32, bias=True).cuda()
    ours = CudaLinear(64, 32, bias=True).cuda()
    ours.weight.data.copy_(ref.weight.data)
    ours.bias.data.copy_(ref.bias.data)
    x = torch.randn(8, 16, 64, device="cuda")
    torch.testing.assert_close(ours(x), ref(x), atol=1e-3, rtol=1e-3)


@cuda_only
def test_cuda_embedding_matches_nn_embedding():
    from cuda_qwen3_vl.modules import CudaEmbedding
    torch.manual_seed(0)
    ref = torch.nn.Embedding(100, 64).cuda()
    ours = CudaEmbedding(100, 64).cuda()
    ours.weight.data.copy_(ref.weight.data)
    ids = torch.randint(0, 100, (4, 16), device="cuda", dtype=torch.int64)
    torch.testing.assert_close(ours(ids), ref(ids), atol=1e-5, rtol=1e-5)


@cuda_only
def test_cuda_rmsnorm():
    from cuda_qwen3_vl.modules import CudaRMSNorm
    torch.manual_seed(0)
    norm = CudaRMSNorm(256, eps=1e-6).cuda()
    x = torch.randn(4, 16, 256, device="cuda")
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight
    torch.testing.assert_close(norm(x), expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_cuda_layernorm_matches_nn_layernorm():
    from cuda_qwen3_vl.modules import CudaLayerNorm
    torch.manual_seed(0)
    ref = torch.nn.LayerNorm(256, eps=1e-6).cuda()
    ours = CudaLayerNorm(256, eps=1e-6).cuda()
    ours.weight.data.copy_(ref.weight.data)
    ours.bias.data.copy_(ref.bias.data)
    x = torch.randn(4, 16, 256, device="cuda")
    torch.testing.assert_close(ours(x), ref(x), atol=1e-4, rtol=1e-4)


@cuda_only
def test_swiglu_mlp_forward_finite():
    """MLP forward produces finite outputs for a realistic small config."""
    from cuda_qwen3_vl.modules import CudaSwiGLUMLP
    torch.manual_seed(0)
    mlp = CudaSwiGLUMLP(hidden_size=64, intermediate_size=128).cuda()
    x = torch.randn(2, 8, 64, device="cuda")
    out = mlp(x)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()


@cuda_only
def test_vision_block_forward_finite():
    from cuda_qwen3_vl.modules import CudaVisionBlock
    torch.manual_seed(0)
    block = CudaVisionBlock(hidden_size=64, num_heads=4, intermediate_size=128).cuda()
    x = torch.randn(1, 32, 64, device="cuda")
    head_dim = 64 // 4
    cos = torch.randn(1, 32, head_dim, device="cuda")
    sin = torch.randn(1, 32, head_dim, device="cuda")
    out = block(x, cos, sin)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@cuda_only
def test_moe_backward():
    """Backward through a full MoE step produces finite grads on input, gate, and experts."""
    from cuda_qwen3_vl.modules import CudaSparseMoE
    torch.manual_seed(0)
    moe = CudaSparseMoE(
        hidden_size=64, moe_intermediate_size=32,
        num_experts=4, top_k=2, norm_topk_prob=True,
    ).cuda()
    x = torch.randn(2, 8, 64, device="cuda", requires_grad=True)
    out, _ = moe(x)
    out.sum().backward()
    assert torch.isfinite(x.grad).all(), "input grad has NaN/Inf"
    assert torch.isfinite(moe.gate_weight.grad).all(), "gate_weight grad has NaN/Inf"
    assert torch.isfinite(moe.gate_up_proj.grad).all(), "gate_up_proj grad has NaN/Inf"
    assert torch.isfinite(moe.down_proj.grad).all(), "down_proj grad has NaN/Inf"
    # Non-trivial: at least some experts must have non-zero grad
    assert moe.gate_up_proj.grad.abs().sum().item() > 0
    assert moe.down_proj.grad.abs().sum().item() > 0


@cuda_only
def test_moe_routing_autograd_wrappers():
    """cuda_topk, cuda_index_add, cuda_batched_gemm all have working backward paths."""
    from cuda_qwen3_vl.kernels import cuda_batched_gemm, cuda_index_add, cuda_topk

    # topk backward: grad-vals scatter back to the selected positions
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    v, _ = cuda_topk(x, 3)
    v.sum().backward()
    assert torch.isfinite(x.grad).all()
    # Only the top-3 positions per row should have non-zero grad
    assert (x.grad != 0).sum(dim=-1).eq(3).all()

    # index_add backward: grad flows to target (identity) and source (gather by index)
    tgt = torch.zeros(6, 16, device="cuda", requires_grad=True)
    src = torch.randn(3, 16, device="cuda", requires_grad=True)
    idx = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int64)
    out = cuda_index_add(tgt, src, idx)
    out.sum().backward()
    torch.testing.assert_close(tgt.grad, torch.ones_like(tgt))
    torch.testing.assert_close(src.grad, torch.ones_like(src))

    # batched_gemm backward: gradients w.r.t. both x and w via the einsum-fallback bwd
    x = torch.randn(3, 8, 16, device="cuda", requires_grad=True)
    w = torch.randn(3, 12, 16, device="cuda", requires_grad=True)
    y = cuda_batched_gemm(x, w)
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
    assert x.grad.abs().sum().item() > 0 and w.grad.abs().sum().item() > 0


@cuda_only
def test_moe_forward_finite():
    """MoE forward produces finite outputs with correct shape."""
    from cuda_qwen3_vl.modules import CudaSparseMoE
    torch.manual_seed(0)
    moe = CudaSparseMoE(
        hidden_size=64, moe_intermediate_size=32,
        num_experts=4, top_k=2, norm_topk_prob=True,
    ).cuda()
    x = torch.randn(2, 8, 64, device="cuda")
    out, router_logits = moe(x)
    assert out.shape == (2, 8, 64)
    assert router_logits.shape == (16, 4)
    assert torch.isfinite(out).all()
