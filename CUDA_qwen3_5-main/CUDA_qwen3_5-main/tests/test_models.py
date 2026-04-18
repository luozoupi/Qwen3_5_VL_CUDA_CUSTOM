"""End-to-end sanity test for the dense and MoE model skeletons with tiny configs."""
import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _tiny_dense_config():
    from cuda_qwen3_vl.configs import Qwen3VLConfig, VisionConfig, TextConfig
    return Qwen3VLConfig(
        family="dense",
        vision=VisionConfig(
            hidden_size=64, num_layers=2, num_heads=4, intermediate_size=128,
            patch_size=8, temporal_patch_size=2, spatial_merge_size=2,
            in_channels=3, num_position_embeddings=256, out_hidden_size=64,
            rope_theta=10000.0, rms_norm_eps=1e-6, deepstack_layers=(),
        ),
        text=TextConfig(
            hidden_size=64, intermediate_size=128, num_layers=2, num_heads=4, num_kv_heads=2,
            head_dim=16, vocab_size=256, max_position_embeddings=64,
            rope_theta=10000.0, rms_norm_eps=1e-6, mrope_section=[6, 5, 5],
            attention_bias=False, tie_word_embeddings=False,
        ),
    )


def _tiny_moe_config():
    from cuda_qwen3_vl.configs import Qwen3VLConfig, VisionConfig, MoETextConfig
    return Qwen3VLConfig(
        family="moe",
        vision=VisionConfig(
            hidden_size=64, num_layers=2, num_heads=4, intermediate_size=128,
            patch_size=8, temporal_patch_size=2, spatial_merge_size=2,
            in_channels=3, num_position_embeddings=256, out_hidden_size=64,
            rope_theta=10000.0, rms_norm_eps=1e-6, deepstack_layers=(),
        ),
        text=MoETextConfig(
            hidden_size=64, intermediate_size=128, num_layers=2, num_heads=4, num_kv_heads=2,
            head_dim=16, vocab_size=256, max_position_embeddings=64,
            rope_theta=10000.0, rms_norm_eps=1e-6, mrope_section=[6, 5, 5],
            attention_bias=False, tie_word_embeddings=False,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            norm_topk_prob=True, decoder_sparse_step=1,
        ),
    )


@cuda_only
def test_dense_model_forward_text_only():
    from cuda_qwen3_vl.models import CudaQwen3VLDenseModel
    torch.manual_seed(0)
    cfg = _tiny_dense_config()
    model = CudaQwen3VLDenseModel(cfg).cuda()
    input_ids = torch.randint(0, cfg.text.vocab_size, (2, 16), device="cuda", dtype=torch.int64)
    logits = model(input_ids=input_ids)
    assert logits.shape == (2, 16, cfg.text.vocab_size)
    assert torch.isfinite(logits).all()


@cuda_only
def test_dense_model_batch_size_2():
    """B=2 should produce finite logits with shape (2, S, V)."""
    from cuda_qwen3_vl.models import CudaQwen3VLDenseModel
    torch.manual_seed(0)
    cfg = _tiny_dense_config()
    model = CudaQwen3VLDenseModel(cfg).cuda()
    input_ids = torch.randint(0, cfg.text.vocab_size, (2, 16), device="cuda", dtype=torch.int64)
    logits = model(input_ids=input_ids)
    assert logits.shape == (2, 16, cfg.text.vocab_size)
    assert torch.isfinite(logits).all()


@cuda_only
def test_dense_model_batch_independence():
    """Per-sample logits in a batch should match the same sample run alone."""
    from cuda_qwen3_vl.models import CudaQwen3VLDenseModel
    torch.manual_seed(0)
    cfg = _tiny_dense_config()
    model = CudaQwen3VLDenseModel(cfg).cuda().eval()
    ids_a = torch.randint(0, cfg.text.vocab_size, (1, 12), device="cuda", dtype=torch.int64)
    ids_b = torch.randint(0, cfg.text.vocab_size, (1, 12), device="cuda", dtype=torch.int64)
    ids_batch = torch.cat([ids_a, ids_b], dim=0)
    with torch.no_grad():
        out_a = model(input_ids=ids_a)
        out_b = model(input_ids=ids_b)
        out_batch = model(input_ids=ids_batch)
    torch.testing.assert_close(out_batch[0:1], out_a, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out_batch[1:2], out_b, atol=1e-3, rtol=1e-3)


@cuda_only
def test_moe_model_forward_text_only():
    from cuda_qwen3_vl.models import CudaQwen3VLMoeModel
    torch.manual_seed(0)
    cfg = _tiny_moe_config()
    model = CudaQwen3VLMoeModel(cfg).cuda()
    input_ids = torch.randint(0, cfg.text.vocab_size, (2, 16), device="cuda", dtype=torch.int64)
    logits, router_logits = model(input_ids=input_ids)
    assert logits.shape == (2, 16, cfg.text.vocab_size)
    assert torch.isfinite(logits).all()
    assert len(router_logits) == cfg.text.num_layers
