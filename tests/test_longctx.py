"""tests/test_longctx.py — Tests for NanoMind long-context efficient attention."""
import math
import pytest
import torch
from nanomind.longctx import (
    RotaryEmbedding, ALiBi, SlidingWindowAttention,
    LinearAttention, RetNetDecay, GroupedQueryAttention,
    ChunkedAttention, LongContextConfig, LongContextLM,
)

D, V, H = 32, 16, 4


# ── RotaryEmbedding ───────────────────────────────────────────────────────────

class TestRoPE:
    def _rope(self, dim=D):
        return RotaryEmbedding(dim=dim, max_seq=64)

    def test_output_shape(self):
        rope = self._rope()
        q    = torch.randn(2, H, 8, D)
        k    = torch.randn(2, H, 8, D)
        q_r, k_r = rope.apply(q, k, seq_len=8)
        assert q_r.shape == q.shape
        assert k_r.shape == k.shape

    def test_different_positions_different_output(self):
        rope = self._rope()
        q = torch.randn(1, 1, 4, D)
        k = torch.randn(1, 1, 4, D)
        q1, _ = rope.apply(q, k, seq_len=4)
        q2, _ = rope.apply(q, k, seq_len=4, offset=4)
        assert not torch.allclose(q1, q2)

    def test_norm_preserved(self):
        rope = self._rope()
        q = torch.randn(2, H, 4, D)
        k = torch.randn(2, H, 4, D)
        q_r, _ = rope.apply(q, k, seq_len=4)
        # RoPE is a rotation — should preserve norm
        assert torch.allclose(q.norm(dim=-1), q_r.norm(dim=-1), atol=1e-5)

    def test_ntk_scaling(self):
        rope = RotaryEmbedding(D, scale_factor=4.0, scaling_type="ntk", max_seq=64)
        q = torch.randn(1, 1, 4, D)
        k = torch.randn(1, 1, 4, D)
        q_r, _ = rope.apply(q, k, seq_len=4)
        assert q_r.shape == q.shape

    def test_extend_cache(self):
        rope = RotaryEmbedding(D, max_seq=32)
        rope.extend(64)
        assert rope.max_seq == 64

    def test_to_dict_keys(self):
        d = self._rope().to_dict()
        for k in ("dim", "base", "max_seq"):
            assert k in d


# ── ALiBi ─────────────────────────────────────────────────────────────────────

class TestALiBi:
    def test_bias_shape(self):
        a    = ALiBi(n_heads=H, max_seq=32)
        bias = a.bias(seq_len=8)
        assert bias.shape == (H, 8, 8)

    def test_causal_masking(self):
        a    = ALiBi(n_heads=H)
        bias = a.bias(seq_len=8)
        # Upper triangle should be -inf
        for h in range(H):
            assert bias[h, 0, 1].item() == float("-inf")

    def test_diagonal_is_zero(self):
        a    = ALiBi(n_heads=H)
        bias = a.bias(seq_len=8)
        for h in range(H):
            assert bias[h, 5, 5].item() == 0.0

    def test_slopes_positive(self):
        a = ALiBi(n_heads=H)
        assert (a.slopes > 0).all()

    def test_apply_to_scores_shape(self):
        a      = ALiBi(n_heads=H)
        scores = torch.randn(2, H, 8, 8)
        out    = a.apply_to_scores(scores, seq_len=8)
        assert out.shape == scores.shape

    def test_n_slopes_equals_n_heads(self):
        for n in [4, 8, 12]:
            a = ALiBi(n_heads=n)
            assert len(a.slopes) == n


# ── SlidingWindowAttention ────────────────────────────────────────────────────

class TestSWA:
    def _swa(self, W=8, sinks=2):
        return SlidingWindowAttention(D, H, window_size=W, n_sinks=sinks)

    def test_output_shape(self):
        swa  = self._swa()
        x    = torch.randn(2, 16, D)
        out, kv = swa(x)
        assert out.shape == (2, 16, D)

    def test_kv_cache_shape(self):
        swa  = self._swa()
        x    = torch.randn(2, 8, D)
        _, (k, v) = swa(x)
        assert k.shape[2] == 8

    def test_gradient_flows(self):
        swa = self._swa()
        x   = torch.randn(2, 4, D, requires_grad=True)
        out, _ = swa(x)
        out.sum().backward()
        assert x.grad is not None

    def test_effective_context(self):
        swa = SlidingWindowAttention(D, H, window_size=64, n_sinks=4)
        assert swa.effective_context(8) == 8 * 64 + 4


# ── LinearAttention ───────────────────────────────────────────────────────────

class TestLinearAttention:
    def test_output_shape(self):
        la  = LinearAttention(D, H, feature="elu")
        x   = torch.randn(2, 6, D)
        assert la(x).shape == (2, 6, D)

    def test_relu_feature(self):
        la  = LinearAttention(D, H, feature="relu")
        x   = torch.randn(2, 4, D)
        assert la(x).shape == (2, 4, D)

    def test_gradient_flows(self):
        la = LinearAttention(D, H)
        x  = torch.randn(2, 4, D, requires_grad=True)
        la(x).sum().backward()
        assert x.grad is not None

    def test_complexity_string(self):
        la = LinearAttention(D, H)
        assert "linear" in la.complexity.lower() or "T" in la.complexity


# ── RetNetDecay ───────────────────────────────────────────────────────────────

class TestRetNetDecay:
    def test_output_shape(self):
        r = RetNetDecay(D, H)
        x = torch.randn(2, 6, D)
        assert r(x).shape == (2, 6, D)

    def test_gamma_range(self):
        r = RetNetDecay(D, H, gamma_min=0.8, gamma_max=0.99)
        assert r.gammas.min().item() >= 0.8
        assert r.gammas.max().item() <= 0.99


# ── GroupedQueryAttention ──────────────────────────────────────────────────────

class TestGQA:
    def test_mha_output_shape(self):
        gqa = GroupedQueryAttention(D, H, n_kv_heads=H, max_seq=32)
        x   = torch.randn(2, 8, D)
        out, kv = gqa(x)
        assert out.shape == (2, 8, D)

    def test_gqa_output_shape(self):
        gqa = GroupedQueryAttention(D, H, n_kv_heads=2, max_seq=32)
        x   = torch.randn(2, 8, D)
        out, _ = gqa(x)
        assert out.shape == (2, 8, D)

    def test_mqa_output_shape(self):
        gqa = GroupedQueryAttention(D, H, n_kv_heads=1, max_seq=32)
        x   = torch.randn(2, 8, D)
        out, _ = gqa(x)
        assert out.shape == (2, 8, D)

    def test_kv_cache_factor(self):
        gqa = GroupedQueryAttention(D, H, n_kv_heads=2)
        assert gqa.kv_cache_factor == 0.5

    def test_to_dict_keys(self):
        d = GroupedQueryAttention(D, H, n_kv_heads=2).to_dict()
        for k in ("n_heads", "n_kv_heads", "kv_reduction"):
            assert k in d

    def test_invalid_kv_heads_raises(self):
        with pytest.raises(AssertionError):
            GroupedQueryAttention(D, n_heads=4, n_kv_heads=3)


# ── ChunkedAttention ──────────────────────────────────────────────────────────

class TestChunkedAttention:
    def test_output_shape(self):
        ca  = ChunkedAttention(D, H, chunk_size=4)
        x   = torch.randn(2, 8, D)
        assert ca(x).shape == (2, 8, D)

    def test_causal_output_differs_from_noncausal(self):
        x   = torch.randn(2, 8, D)
        ca  = ChunkedAttention(D, H, chunk_size=4, causal=True)
        nc  = ChunkedAttention(D, H, chunk_size=4, causal=False)
        nc.load_state_dict(ca.state_dict())
        o1  = ca(x)
        o2  = nc(x)
        assert not torch.allclose(o1, o2, atol=1e-3)

    def test_gradient_flows(self):
        ca = ChunkedAttention(D, H, chunk_size=4)
        x  = torch.randn(2, 4, D, requires_grad=True)
        ca(x).sum().backward()
        assert x.grad is not None


# ── LongContextLM ─────────────────────────────────────────────────────────────

class TestLongContextLM:
    def _model(self, attn="gqa", pos="rope"):
        cfg = LongContextConfig(V, D, n_layers=1, n_heads=H, n_kv_heads=2,
                                 max_seq=16, window_size=8, attn_type=attn,
                                 pos_encoding=pos)
        return LongContextLM(cfg)

    def test_gqa_rope_logits(self):
        m   = self._model("gqa", "rope")
        ids = torch.randint(0, V, (2, 6))
        logits, _ = m(ids)
        assert logits.shape == (2, 6, V)

    def test_sliding_alibi_logits(self):
        m   = self._model("sliding", "alibi")
        ids = torch.randint(0, V, (2, 6))
        logits, _ = m(ids)
        assert logits.shape == (2, 6, V)

    def test_linear_none_logits(self):
        m   = self._model("linear", "none")
        ids = torch.randint(0, V, (2, 6))
        logits, _ = m(ids)
        assert logits.shape == (2, 6, V)

    def test_loss_scalar(self):
        m   = self._model()
        ids = torch.randint(0, V, (2, 6))
        _, loss = m(ids, ids)
        assert loss.shape == ()

    def test_n_params_positive(self):
        m = self._model()
        assert m.n_params > 0

    def test_gradient_flows(self):
        m    = self._model()
        ids  = torch.randint(0, V, (2, 4))
        _, l = m(ids, ids)
        l.backward()
        has_grad = any(p.grad is not None for p in m.parameters())
        assert has_grad


class TestRoPEScaling:
    def test_linear_scaling_changes_freqs(self):
        r1 = RotaryEmbedding(D, scale_factor=1.0, scaling_type="none", max_seq=32)
        r2 = RotaryEmbedding(D, scale_factor=4.0, scaling_type="linear", max_seq=32)
        q = torch.randn(1, 1, 4, D); k = torch.randn(1, 1, 4, D)
        q1, _ = r1.apply(q, k, seq_len=4)
        q2, _ = r2.apply(q, k, seq_len=4)
        assert not torch.allclose(q1, q2)


class TestALiBiCache:
    def test_bias_cached(self):
        a  = ALiBi(n_heads=H)
        b1 = a.bias(16)
        b2 = a.bias(16)
        assert b1 is b2   # should be same object from cache

    def test_different_lengths_different_bias(self):
        a  = ALiBi(n_heads=H)
        b1 = a.bias(8)
        b2 = a.bias(16)
        assert b1.shape != b2.shape
