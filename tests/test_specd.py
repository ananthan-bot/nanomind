"""tests/test_specd.py — Tests for NanoMind speculative decoding package."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.specd import (
    NgramDraftModel, SmallModelDraft,
    SpeculativeSampler, SpeculativeResult,
    SpeculativeDecoder, GenerationStats,
    MedusaHead, MedusaModel,
    TokenTree, TreeNode,
    LookaheadDecoder,
)

V = 16


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 8)
        self.rnn  = nn.GRU(8, 16, batch_first=True)
        self.head = nn.Linear(16, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None


# ── NgramDraftModel ───────────────────────────────────────────────────────────

class TestNgramDraft:
    def _draft(self):
        d = NgramDraftModel(vocab_size=V, n=2)
        d.train_ngrams([[0, 1, 2, 3, 4, 5]] * 5)
        return d

    def test_draft_shape(self):
        d   = self._draft()
        ids = torch.tensor([[0, 1, 2]])
        di, dl = d.draft(ids, n_tokens=3)
        assert di.shape == (1, 3)
        assert dl.shape == (1, 3, V)

    def test_draft_ids_in_range(self):
        d   = self._draft()
        ids = torch.tensor([[0, 1]])
        di, _ = d.draft(ids, n_tokens=4)
        assert (di >= 0).all() and (di < V).all()

    def test_logits_shape(self):
        d  = self._draft()
        l  = d.logits(torch.tensor([[0, 1, 2]]))
        assert l.shape == (1, V)


# ── SmallModelDraft ───────────────────────────────────────────────────────────

class TestSmallModelDraft:
    def test_draft_shape(self):
        d   = SmallModelDraft(TinyLM())
        ids = torch.randint(0, V, (1, 4))
        di, dl = d.draft(ids, n_tokens=3)
        assert di.shape == (1, 3)
        assert dl.shape == (1, 3, V)

    def test_logits_shape(self):
        d   = SmallModelDraft(TinyLM())
        ids = torch.randint(0, V, (2, 5))
        l   = d.logits(ids)
        assert l.shape == (2, V)


# ── SpeculativeSampler ────────────────────────────────────────────────────────

class TestSpeculativeSampler:
    def _s(self):
        return SpeculativeSampler(temperature=1.0)

    def test_verify_output_shape(self):
        s = self._s()
        K = 4
        di = torch.randint(0, V, (2, K))
        dl = torch.randn(2, K, V)
        tl = torch.randn(2, K + 1, V)
        r  = s.verify(di, dl, tl)
        assert r.accepted_ids.shape[0] == 2
        assert r.accepted_ids.shape[1] >= 1

    def test_acceptance_rate_in_range(self):
        s  = self._s()
        di = torch.randint(0, V, (1, 4))
        dl = torch.randn(1, 4, V)
        tl = torch.randn(1, 5, V)
        r  = s.verify(di, dl, tl)
        assert 0.0 <= r.acceptance_rate <= 1.0

    def test_greedy_verify_shape(self):
        s  = self._s()
        di = torch.randint(0, V, (1, 4))
        tl = torch.randn(1, 5, V)
        r  = s.greedy_verify(di, tl)
        assert r.accepted_ids.shape[0] == 1

    def test_efficiency_gain(self):
        r = SpeculativeResult(
            accepted_ids=torch.zeros(1, 3, dtype=torch.long),
            n_accepted=3, acceptance_rate=0.75,
        )
        assert r.efficiency_gain(4) == 3.0


# ── SpeculativeDecoder ────────────────────────────────────────────────────────

class TestSpeculativeDecoder:
    def _engine(self):
        ngram = NgramDraftModel(V, n=2)
        ngram.train_ngrams([[i % V for i in range(20)]])
        return SpeculativeDecoder(TinyLM(), ngram, k=3, temperature=1.0)

    def test_generate_shape(self):
        e   = self._engine()
        ids = torch.randint(0, V, (1, 4))
        out = e.generate(ids, max_new_tokens=6)
        assert out.shape[1] >= ids.shape[1] + 6

    def test_stats_populated(self):
        e   = self._engine()
        ids = torch.randint(0, V, (1, 4))
        e.generate(ids, max_new_tokens=4)
        s = e.stats
        assert s.n_target_calls > 0
        assert s.n_tokens_generated > 0

    def test_speedup_positive(self):
        e   = self._engine()
        ids = torch.randint(0, V, (1, 4))
        e.generate(ids, max_new_tokens=6)
        assert e.stats.speedup > 0

    def test_naive_generate_shape(self):
        e   = self._engine()
        ids = torch.randint(0, V, (1, 4))
        out = e.generate_naive(ids, max_new_tokens=4)
        assert out.shape == (1, 8)


# ── MedusaModel ───────────────────────────────────────────────────────────────

class TestMedusa:
    def _model(self):
        return MedusaModel(TinyLM(), n_heads=2, d_model=V, vocab_size=V)

    def test_forward_shapes(self):
        m   = self._model()
        ids = torch.randint(0, V, (2, 6))
        bl, ml = m(ids)
        assert bl.shape == (2, 6, V)
        assert len(ml) == 2
        assert ml[0].shape == (2, 6, V)

    def test_speculate_shapes(self):
        m   = self._model()
        ids = torch.randint(0, V, (1, 4))
        bt, mt = m.speculate(ids, top_k_each=2)
        assert bt.shape == (1, 2)
        assert len(mt) == 2

    def test_medusa_loss_scalars(self):
        m    = self._model()
        ids  = torch.randint(0, V, (2, 6))
        lbl  = torch.randint(0, V, (2, 6))
        bl, ml = m.medusa_loss(ids, lbl)
        assert bl.shape == ()
        assert all(l.shape == () for l in ml)

    def test_n_medusa_params_positive(self):
        m = self._model()
        assert m.n_medusa_params > 0

    def test_total_params_gt_base(self):
        base  = TinyLM()
        m     = MedusaModel(base, n_heads=2, d_model=V, vocab_size=V)
        base_p = sum(p.numel() for p in base.parameters())
        assert m.n_params > base_p


# ── TokenTree ─────────────────────────────────────────────────────────────────

class TestTokenTree:
    def test_build_and_paths(self):
        tree  = TokenTree(max_depth=2, branching=2)
        cands = [[1, 2], [3, 4]]
        root  = tree.build(cands)
        paths = tree.all_paths(root)
        assert len(paths) > 0
        for p in paths:
            assert len(p) > 0

    def test_n_nodes(self):
        tree  = TokenTree(max_depth=2, branching=2)
        root  = tree.build([[1, 2], [3, 4]])
        n     = tree.n_nodes(root)
        assert n > 1   # at least root + children

    def test_verify_paths(self):
        tree   = TokenTree(max_depth=2, branching=2)
        root   = tree.build([[5, 7], [3, 8]])
        target = [5, 3]
        best   = tree.verify_paths(root, target)
        assert isinstance(best, list)

    def test_empty_cands(self):
        tree = TokenTree(max_depth=2, branching=2)
        root = tree.build([])
        assert root.token_id == -1


# ── LookaheadDecoder ──────────────────────────────────────────────────────────

class TestLookaheadDecoder:
    def test_generate_shape(self):
        d   = LookaheadDecoder(TinyLM(), window_size=3, n_iters=1)
        ids = torch.randint(0, V, (1, 4))
        out = d.generate(ids, max_new_tokens=4)
        assert out.shape == (1, 8)

    def test_longer_than_input(self):
        d   = LookaheadDecoder(TinyLM(), window_size=4, n_iters=2)
        ids = torch.randint(0, V, (1, 5))
        out = d.generate(ids, max_new_tokens=6)
        assert out.shape[1] > ids.shape[1]


class TestNgramUntrained:
    def test_untrained_repeats_last(self):
        d   = NgramDraftModel(vocab_size=V, n=2)
        ids = torch.tensor([[5, 3]])
        di, _ = d.draft(ids, n_tokens=3)
        # Without training, should default to repeating last token (3)
        assert di[0, 0].item() == 3


class TestGreedySampler:
    def test_low_temperature_deterministic(self):
        s  = SpeculativeSampler(temperature=0.01)
        di = torch.randint(0, V, (1, 3))
        dl = torch.randn(1, 3, V)
        tl = torch.randn(1, 4, V)
        r1 = s.verify(di, dl, tl)
        r2 = s.verify(di, dl, tl)
        # Very low temp → nearly deterministic
        assert r1.accepted_ids.shape == r2.accepted_ids.shape


class TestSpecDecoderK1:
    def test_k1_generates_tokens(self):
        ngram = NgramDraftModel(V, n=1)
        ngram.train_ngrams([[i % V for i in range(16)]])
        engine = SpeculativeDecoder(TinyLM(), ngram, k=1)
        ids    = torch.randint(0, V, (1, 4))
        out    = engine.generate(ids, max_new_tokens=4)
        assert out.shape[1] >= ids.shape[1] + 4


class TestGenerationStats:
    def test_to_dict_keys(self):
        s = GenerationStats(n_tokens_generated=20, n_target_calls=5,
                             total_accepted=18, total_drafted=20,
                             acceptance_rates=[0.9, 0.85])
        d = s.to_dict()
        for k in ("tokens_generated", "target_model_calls",
                   "mean_accepted_per_step", "acceptance_rate", "speedup_factor"):
            assert k in d
