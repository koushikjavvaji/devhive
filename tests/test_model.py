import torch

from model.attention import GroupedQueryAttention
from model.block import TransformerBlock
from model.config import GPTConfig
from model.embeddings import TokenEmbedding
from model.feedforward import SwiGLU
from model.gpt import GPT
from model.norm import RMSNorm
from model.rope import RoPE


def tiny_config(**overrides):
    defaults = dict(vocab_size=50, d_model=16, n_layers=3, n_heads=4, n_kv_heads=2, d_ff=32, max_seq_len=20)
    defaults.update(overrides)
    return GPTConfig(**defaults)


def test_token_embedding_shape():
    embedding = TokenEmbedding(vocab_size=10, d_model=4)
    out = embedding(torch.tensor([[1, 2, 3]]))
    assert out.shape == (1, 3, 4)


def test_rmsnorm_normalizes_scale():
    norm = RMSNorm(d_model=8)
    x = torch.randn(4, 8) * 10
    rms = norm(x).pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones(4), atol=1e-4)


def test_rope_preserves_vector_norm():
    # rotation is norm-preserving per (even, odd) pair, so the norm at each
    # position should be unchanged by RoPE regardless of position
    rope = RoPE(d_model=8, max_seq_len=16)
    x = torch.randn(1, 1, 5, 8)
    rotated = rope(x)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)


def test_swiglu_preserves_shape():
    ffn = SwiGLU(d_model=8, d_ff=32)
    x = torch.randn(2, 5, 8)
    assert ffn(x).shape == x.shape


def test_gqa_keeps_kv_heads_ungrouped():
    """Regression test: GQA previously repeat_interleave'd k/v up to n_heads before
    attending, which defeats the whole memory-saving point of grouped-query attention."""
    attn = GroupedQueryAttention(d_model=8, n_heads=4, n_kv_heads=2)
    x = torch.randn(2, 5, 8)
    out, (k, v) = attn(x)

    assert out.shape == x.shape
    assert k.shape[1] == 2
    assert v.shape[1] == 2


def test_transformer_block_preserves_shape():
    block = TransformerBlock(d_model=8, n_heads=4, n_kv_heads=2, d_ff=32)
    x = torch.randn(2, 5, 8)
    out, _ = block(x)
    assert out.shape == x.shape


def test_gpt_forward_and_loss():
    config = tiny_config()
    model = GPT(config)
    tokens = torch.randint(0, config.vocab_size, (2, 10))
    targets = torch.randint(0, config.vocab_size, (2, 10))

    logits, loss, kv_caches = model(tokens, targets)

    assert logits.shape == (2, 10, config.vocab_size)
    assert loss.item() > 0
    assert len(kv_caches) == config.n_layers


def test_gpt_ties_output_head_to_embedding():
    model = GPT(tiny_config())
    assert model.lm_head.weight is model.token_emb.weight


def test_kv_cache_matches_full_recompute():
    """Regression test for a real bug: F.scaled_dot_product_attention(is_causal=True)
    top-left-aligns its mask when q is shorter than k (this torch version), so a naive
    decode step would only ever attend to the very first cached token instead of the
    whole cache. This pins decode-with-cache against a full from-scratch forward pass."""
    torch.manual_seed(0)
    config = tiny_config()
    model = GPT(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 12))

    with torch.no_grad():
        full_logits, _, _ = model(tokens, start_pos=0)

        prefill = tokens[:, :5]
        logits, _, kv_caches = model(prefill, start_pos=0)
        cached_logits = [logits[:, -1, :]]

        for pos in range(5, 12):
            tok = tokens[:, pos:pos + 1]
            logits, _, kv_caches = model(tok, start_pos=pos, kv_caches=kv_caches)
            cached_logits.append(logits[:, -1, :])

    cached_logits = torch.cat(cached_logits, dim=0)
    full_at_those_positions = full_logits[0, 4:12, :]

    assert torch.allclose(cached_logits, full_at_those_positions, atol=1e-4)
