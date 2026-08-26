"""
tests/test_speculative.py — Tests for speculative decoding.
"""

import pytest
import torch

from nanomind import NanoMind, ModelConfig
from nanomind.speculative import (
    SpeculativeConfig,
    SpeculativeGenerator,
    speculative_decode,
    generate_draft,
    verify_draft,
    rejection_sample,
    SpeculativeStats,
)
from nanomind.tokenizer.char import CharTokenizer

VOCAB  = 32
BLOCK  = 16
D_BIG  = 64
D_SML  = 32
B      = 1

def big_model():
    torch.manual_seed(0)
    return NanoMind(ModelConfig(vocab_size=VOCAB, block_size=BLOCK,
                                d_model=D_BIG, n_layers=2, n_heads=4, dropout=0.0))

def small_model():
    torch.manual_seed(1)
    return NanoMind(ModelConfig(vocab_size=VOCAB, block_size=BLOCK,
                                d_model=D_SML, n_layers=1, n_heads=2, dropout=0.0))

TOKENIZER = CharTokenizer().build("abcdefghijklmnopqrstuvwxyz " * 5)

def make_idx(t=4):
    return torch.randint(0, VOCAB, (1, t))


# ── SpeculativeConfig ─────────────────────────────────────────────────────────

class TestSpeculativeConfig:
    def test_defaults(self):
        cfg = SpeculativeConfig()
        assert cfg.n_draft == 5
        assert cfg.max_new_tokens == 100

    def test_invalid_n_draft(self):
        with pytest.raises(AssertionError):
            SpeculativeConfig(n_draft=0)

    def test_scaling_property(self):
        cfg = SpeculativeConfig(temperature=0.5)
        assert cfg.temperature == 0.5


# ── generate_draft ────────────────────────────────────────────────────────────

class TestGenerateDraft:
    def test_output_shapes(self):
        model = small_model()
        idx   = make_idx()
        ids, probs = generate_draft(model, idx, n_draft=5)
        assert ids.shape   == (5,)
        assert probs.shape == (5,)

    def test_probs_in_range(self):
        model = small_model()
        idx   = make_idx()
        _, probs = generate_draft(model, idx, n_draft=5)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_ids_in_vocab(self):
        model = small_model()
        idx   = make_idx()
        ids, _ = generate_draft(model, idx, n_draft=5)
        assert (ids >= 0).all() and (ids < VOCAB).all()

    def test_n_draft_respected(self):
        model = small_model()
        idx   = make_idx()
        for n in [1, 3, 8]:
            ids, probs = generate_draft(model, idx, n_draft=n)
            assert ids.shape[0] == n


# ── verify_draft ──────────────────────────────────────────────────────────────

class TestVerifyDraft:
    def test_output_shapes(self):
        target = big_model()
        idx    = make_idx(4)
        draft_ids, _ = generate_draft(small_model(), idx, n_draft=3)
        probs_at_draft, all_logits = verify_draft(target, idx, draft_ids)
        assert probs_at_draft.shape == (3,)
        assert all_logits.shape     == (4, VOCAB)   # 3 draft + 1 bonus

    def test_probs_sum_to_reasonable(self):
        target = big_model()
        idx    = make_idx(4)
        draft_ids = torch.randint(0, VOCAB, (3,))
        probs_at_draft, _ = verify_draft(target, idx, draft_ids)
        assert (probs_at_draft >= 0).all()
        assert (probs_at_draft <= 1).all()

    def test_logits_finite(self):
        target = big_model()
        idx    = make_idx(4)
        draft_ids = torch.randint(0, VOCAB, (3,))
        _, logits = verify_draft(target, idx, draft_ids)
        assert logits.isfinite().all()


# ── rejection_sample ──────────────────────────────────────────────────────────

class TestRejectionSample:
    def _make_inputs(self, n=3):
        draft_ids   = torch.randint(0, VOCAB, (n,))
        draft_probs = torch.rand(n).clamp(1e-4, 1.0)
        target_probs_at_draft = torch.rand(n).clamp(1e-4, 1.0)
        target_logits = torch.randn(n + 1, VOCAB)
        return draft_ids, draft_probs, target_probs_at_draft, target_logits

    def test_output_is_not_empty(self):
        args = self._make_inputs()
        tokens, n_acc = rejection_sample(*args)
        assert len(tokens) >= 1

    def test_n_accepted_bounded(self):
        n = 4
        args = self._make_inputs(n)
        tokens, n_acc = rejection_sample(*args)
        assert 0 <= n_acc <= n

    def test_all_accepted_gives_n_plus_1(self):
        """If all accepted, we get n_draft + 1 tokens (bonus token included)."""
        n = 3
        draft_ids   = torch.randint(0, VOCAB, (n,))
        # Target prob >> draft prob → always accept
        draft_probs          = torch.full((n,), 0.001)
        target_probs_at_draft = torch.full((n,), 1.0)
        target_logits        = torch.randn(n + 1, VOCAB)
        tokens, n_acc = rejection_sample(
            draft_ids, draft_probs, target_probs_at_draft, target_logits
        )
        assert n_acc == n
        assert len(tokens) == n + 1

    def test_output_tokens_in_vocab(self):
        args  = self._make_inputs()
        tokens, _ = rejection_sample(*args)
        assert (tokens >= 0).all() and (tokens < VOCAB).all()


# ── speculative_decode ────────────────────────────────────────────────────────

class TestSpeculateDecode:
    def test_output_longer_than_input(self):
        target = big_model()
        draft  = small_model()
        idx    = make_idx(4)
        cfg    = SpeculativeConfig(n_draft=3, max_new_tokens=10)
        out, stats = speculative_decode(target, draft, idx, cfg)
        assert out.shape[1] > idx.shape[1]

    def test_stats_keys_present(self):
        target = big_model()
        draft  = small_model()
        idx    = make_idx(4)
        cfg    = SpeculativeConfig(n_draft=3, max_new_tokens=5)
        _, stats = speculative_decode(target, draft, idx, cfg)
        for key in ("n_tokens", "n_draft_calls", "acceptance_rate"):
            assert key in stats

    def test_acceptance_rate_in_range(self):
        target = big_model()
        draft  = small_model()
        idx    = make_idx(4)
        cfg    = SpeculativeConfig(n_draft=3, max_new_tokens=15)
        _, stats = speculative_decode(target, draft, idx, cfg)
        assert 0.0 <= stats["acceptance_rate"] <= 1.0

    def test_n_tokens_bounded_by_max(self):
        target = big_model()
        draft  = small_model()
        idx    = make_idx(4)
        cfg    = SpeculativeConfig(n_draft=3, max_new_tokens=10)
        out, stats = speculative_decode(target, draft, idx, cfg)
        assert stats["n_tokens"] <= 10 + cfg.n_draft   # slight overshoot possible
