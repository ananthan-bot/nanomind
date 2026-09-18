"""tests/test_interpret.py — Tests for NanoMind interpretability."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.interpret import (
    AttentionMap, AttentionExtractor,
    SaliencyMap, GradientSaliency,
    LinearProbe, ProbeResult, LayerwiseProber,
    LogitLensResult, LogitLens,
    AblationResult, HeadAblator,
    Attribution, OcclusionAttributor, ShapleyAttributor,
    PatchResult, ActivationPatcher,
)

V = 16

class TinyTF(nn.Module):
    def __init__(self, V=16, D=32, T=8, H=2, L=2):
        super().__init__()
        self.T   = T
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.blocks = nn.ModuleList([
            nn.MultiheadAttention(D, H, batch_first=True)
            for _ in range(L)
        ])
        self.lns  = nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.ln   = nn.LayerNorm(D)
        self.lm   = nn.Linear(D, V, bias=False)
    def forward(self, x, t=None):
        B, S = x.shape
        h = self.tok(x) + self.pos(torch.arange(min(S, self.T)))
        h = h[:, :self.T]
        for attn, ln in zip(self.blocks, self.lns):
            r, _ = attn(h, h, h)
            h    = ln(h + r)
        h = self.ln(h)
        logits = self.lm(h)
        loss   = F.cross_entropy(logits.view(-1, V), t.view(-1)) if t is not None else None
        return logits, loss

MODEL = TinyTF(V=V)


# ── AttentionMap ──────────────────────────────────────────────────────────────

class TestAttentionMap:
    def _map(self):
        weights = torch.rand(1, 2, 4, 4)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return AttentionMap(layer=0, weights=weights)

    def test_n_heads(self):
        m = self._map()
        assert m.n_heads == 2

    def test_seq_len(self):
        m = self._map()
        assert m.seq_len == 4

    def test_head_shape(self):
        m = self._map()
        assert m.head(0).shape == (4, 4)

    def test_mean_head_shape(self):
        m = self._map()
        assert m.mean_head().shape == (4, 4)

    def test_entropy_shape(self):
        m = self._map()
        assert m.entropy().shape == (2,)

    def test_entropy_non_negative(self):
        m = self._map()
        assert (m.entropy() >= 0).all()

    def test_rollout_shape(self):
        m = self._map()
        assert m.rollout().shape == (4, 4)

    def test_to_dict_keys(self):
        m = self._map()
        d = m.to_dict()
        for k in ("layer", "n_heads", "seq_len", "entropy"):
            assert k in d


# ── AttentionExtractor ────────────────────────────────────────────────────────

class TestAttentionExtractor:
    def test_extract_returns_list(self):
        ext  = AttentionExtractor(MODEL)
        x    = torch.randint(0, V, (1, 4))
        maps = ext.extract(x)
        assert isinstance(maps, list)

    def test_hooks_removed_after_extract(self):
        ext = AttentionExtractor(MODEL)
        x   = torch.randint(0, V, (1, 4))
        ext.extract(x)
        assert len(ext._hooks) == 0

    def test_tokens_set(self):
        ext  = AttentionExtractor(MODEL)
        x    = torch.randint(0, V, (1, 4))
        toks = ["a", "b", "c", "d"]
        maps = ext.extract(x, tokens=toks)
        for m in maps:
            assert m.tokens == toks


# ── SaliencyMap ───────────────────────────────────────────────────────────────

class TestSaliencyMap:
    def _sal(self):
        return SaliencyMap(torch.tensor([0.1, 0.5, 0.3, 0.8]),
                           tokens=["a", "b", "c", "d"], method="vanilla")

    def test_normalised_range(self):
        n = self._sal().normalised()
        assert n.min().item() >= 0.0
        assert n.max().item() <= 1.0 + 1e-5

    def test_top_k_tokens(self):
        top = self._sal().top_k_tokens(2)
        assert len(top) == 2
        assert top[0][0] == "d"   # highest score

    def test_to_dict_keys(self):
        d = self._sal().to_dict()
        for k in ("method", "scores", "tokens", "top_5"):
            assert k in d


# ── GradientSaliency ──────────────────────────────────────────────────────────

class TestGradientSaliency:
    def _model(self):
        m = TinyTF(V=V)
        m.train()
        return m

    def test_vanilla_shape(self):
        m   = self._model()
        sal = GradientSaliency(m)
        x   = torch.randint(0, V, (1, 4))
        s   = sal.vanilla(x)
        assert s.scores.shape == (4,)

    def test_gxi_shape(self):
        m   = self._model()
        sal = GradientSaliency(m)
        x   = torch.randint(0, V, (1, 4))
        s   = sal.grad_times_input(x)
        assert s.scores.shape == (4,)

    def test_method_label(self):
        m   = self._model()
        sal = GradientSaliency(m)
        x   = torch.randint(0, V, (1, 4))
        assert sal.vanilla(x).method == "vanilla"
        assert sal.grad_times_input(x).method == "grad_times_input"

    def test_scores_non_negative(self):
        m   = self._model()
        sal = GradientSaliency(m)
        x   = torch.randint(0, V, (1, 4))
        s   = sal.vanilla(x)
        assert (s.scores >= 0).all()


# ── LinearProbe ───────────────────────────────────────────────────────────────

class TestLinearProbe:
    def test_fit_returns_probe_result(self):
        probe = LinearProbe(d_model=16, n_classes=3)
        X     = torch.randn(20, 16)
        y     = torch.randint(0, 3, (20,))
        r     = probe.fit(X[:15], y[:15], X[15:], y[15:], epochs=10)
        assert isinstance(r, ProbeResult)

    def test_accuracy_in_range(self):
        probe = LinearProbe(d_model=16, n_classes=3)
        X     = torch.randn(20, 16)
        y     = torch.randint(0, 3, (20,))
        r     = probe.fit(X[:15], y[:15], X[15:], y[15:], epochs=10)
        assert 0.0 <= r.accuracy <= 1.0

    def test_probe_result_keys(self):
        probe = LinearProbe(d_model=8, n_classes=2)
        X     = torch.randn(10, 8)
        y     = torch.randint(0, 2, (10,))
        r     = probe.fit(X[:7], y[:7], X[7:], y[7:])
        d     = r.to_dict()
        for k in ("layer", "task", "accuracy", "loss"):
            assert k in d

    def test_perfect_separable(self):
        """Linearly separable data should achieve high accuracy."""
        probe = LinearProbe(d_model=4, n_classes=2)
        X0    = torch.ones(10, 4)
        X1    = -torch.ones(10, 4)
        X     = torch.cat([X0, X1])
        y     = torch.cat([torch.zeros(10), torch.ones(10)]).long()
        r     = probe.fit(X[:16], y[:16], X[16:], y[16:], epochs=50)
        assert r.accuracy >= 0.5   # should be well above chance


# ── LogitLens ────────────────────────────────────────────────────────────────

class TestLogitLens:
    def test_analyse_returns_result(self):
        lens = LogitLens(MODEL)
        x    = torch.randint(0, V, (1, 4))
        r    = lens.analyse(x)
        assert isinstance(r, LogitLensResult)

    def test_n_layers_positive(self):
        lens = LogitLens(MODEL)
        x    = torch.randint(0, V, (1, 4))
        r    = lens.analyse(x)
        assert r.n_layers >= 0

    def test_layer_tokens_shape(self):
        lens = LogitLens(MODEL)
        x    = torch.randint(0, V, (1, 4))
        r    = lens.analyse(x)
        for toks in r.layer_tokens:
            assert toks.dim() == 1   # (T,)

    def test_prediction_change_length(self):
        lens = LogitLens(MODEL)
        x    = torch.randint(0, V, (1, 4))
        r    = lens.analyse(x)
        assert len(r.prediction_change()) == max(0, r.n_layers - 1)


# ── HeadAblator ───────────────────────────────────────────────────────────────

class TestHeadAblator:
    def test_ablate_all_returns_list(self):
        abl = HeadAblator(MODEL)
        x   = torch.randint(0, V, (1, 4))
        y   = torch.randint(0, V, (1, 4))
        res = abl.ablate_all(x, y)
        assert isinstance(res, list)

    def test_ablation_result_fields(self):
        abl = HeadAblator(MODEL)
        x   = torch.randint(0, V, (1, 4))
        y   = torch.randint(0, V, (1, 4))
        res = abl.ablate_all(x, y)
        if res:
            r = res[0]
            assert hasattr(r, "layer") and hasattr(r, "head")
            assert hasattr(r, "loss_delta")

    def test_importance_non_negative(self):
        abl = HeadAblator(MODEL)
        x   = torch.randint(0, V, (1, 4))
        y   = torch.randint(0, V, (1, 4))
        for r in abl.ablate_all(x, y):
            assert r.importance >= 0.0

    def test_ablation_result_to_dict(self):
        abl = HeadAblator(MODEL)
        x   = torch.randint(0, V, (1, 4))
        y   = torch.randint(0, V, (1, 4))
        res = abl.ablate_all(x, y)
        if res:
            d = res[0].to_dict()
            assert "layer" in d and "importance" in d


# ── OcclusionAttributor ───────────────────────────────────────────────────────

class TestOcclusionAttributor:
    def test_attribute_returns_attribution(self):
        attr = OcclusionAttributor(MODEL)
        x    = torch.randint(0, V, (1, 4))
        r    = attr.attribute(x, target_pos=-1, target_class=0)
        assert isinstance(r, Attribution)

    def test_scores_length(self):
        attr = OcclusionAttributor(MODEL)
        x    = torch.randint(0, V, (1, 4))
        r    = attr.attribute(x, target_pos=-1, target_class=0)
        assert len(r.scores) == 4

    def test_method_label(self):
        attr = OcclusionAttributor(MODEL)
        x    = torch.randint(0, V, (1, 4))
        r    = attr.attribute(x, target_pos=-1, target_class=0)
        assert r.method == "occlusion"

    def test_normalised_sums_to_one(self):
        attr = OcclusionAttributor(MODEL)
        x    = torch.randint(0, V, (1, 4))
        r    = attr.attribute(x, target_pos=-1, target_class=0)
        n    = r.normalised()
        assert abs(sum(abs(v) for v in n) - 1.0) < 0.01


# ── ShapleyAttributor ─────────────────────────────────────────────────────────

class TestShapleyAttributor:
    def test_attribute_returns_attribution(self):
        shap = ShapleyAttributor(MODEL, n_samples=3)
        x    = torch.randint(0, V, (1, 4))
        r    = shap.attribute(x, target_pos=-1, target_class=0)
        assert isinstance(r, Attribution)

    def test_scores_length(self):
        shap = ShapleyAttributor(MODEL, n_samples=3)
        x    = torch.randint(0, V, (1, 4))
        r    = shap.attribute(x, target_pos=-1, target_class=0)
        assert len(r.scores) == 4

    def test_method_label(self):
        shap = ShapleyAttributor(MODEL, n_samples=3)
        x    = torch.randint(0, V, (1, 4))
        r    = shap.attribute(x, target_pos=-1, target_class=0)
        assert r.method == "shapley"


# ── ActivationPatcher ─────────────────────────────────────────────────────────

class TestActivationPatcher:
    def test_trace_returns_list(self):
        patcher = ActivationPatcher(MODEL)
        x1      = torch.randint(0, V, (1, 4))
        x2      = torch.randint(0, V, (1, 4))
        results = patcher.trace(x1, x2)
        assert isinstance(results, list)

    def test_patch_result_fields(self):
        patcher = ActivationPatcher(MODEL)
        x1      = torch.randint(0, V, (1, 4))
        x2      = torch.randint(0, V, (1, 4))
        results = patcher.trace(x1, x2)
        if results:
            r = results[0]
            assert hasattr(r, "layer") and hasattr(r, "recovery")

    def test_recovery_in_range(self):
        patcher = ActivationPatcher(MODEL)
        x1      = torch.randint(0, V, (1, 4))
        x2      = torch.randint(0, V, (1, 4))
        for r in patcher.trace(x1, x2):
            assert -0.1 <= r.recovery <= 1.1   # approximately [0,1]


class TestAttentionRollout:
    def test_rollout_rows_sum_approx_1(self):
        weights = torch.rand(1, 2, 4, 4)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        m   = AttentionMap(layer=0, weights=weights)
        r   = m.rollout()
        row_sums = r.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(4), atol=1e-4)


class TestSaliencyTokenOrder:
    def test_top_k_order(self):
        scores = torch.tensor([0.1, 0.9, 0.3, 0.5])
        sal    = SaliencyMap(scores, tokens=["a","b","c","d"], method="x")
        top    = sal.top_k_tokens(2)
        assert top[0][0] == "b"   # highest
        assert top[1][0] == "d"   # second


class TestLinearProbeForward:
    def test_forward_shape(self):
        probe = LinearProbe(d_model=8, n_classes=4)
        x     = torch.randn(5, 8)
        out   = probe(x)
        assert out.shape == (5, 4)


class TestAttributionDict:
    def test_to_dict_keys(self):
        attr = Attribution(scores=[0.1, 0.5, 0.3],
                            tokens=["a","b","c"],
                            method="occlusion", baseline=0.7)
        d    = attr.to_dict()
        for k in ("method", "tokens", "scores", "top_3"):
            assert k in d
    def test_top_k_length(self):
        attr = Attribution(scores=[0.1, 0.5, 0.3, 0.8],
                            tokens=["a","b","c","d"],
                            method="x", baseline=0.5)
        assert len(attr.top_k(2)) == 2


class TestPatchResultRecovery:
    def test_recovery_formula(self):
        r = PatchResult(layer=0, position=0,
                         clean_score=0.9, corrupt_score=0.5, patch_score=0.7)
        # recovery = (0.7 - 0.5) / (0.9 - 0.5) = 0.5
        assert abs(r.recovery - 0.5) < 1e-4
    def test_to_dict_keys(self):
        r = PatchResult(0, 0, 0.9, 0.5, 0.7)
        d = r.to_dict()
        assert "recovery" in d and "layer" in d


class TestIntegratedGradients:
    def test_ig_shape(self):
        m   = TinyTF(V=V)
        m.eval()
        sal = GradientSaliency(m)
        x   = torch.randint(0, V, (1, 4))
        s   = sal.integrated_gradients(x, n_steps=5)
        assert s.scores.shape == (4,)
    def test_ig_method_label(self):
        m   = TinyTF(V=V)
        sal = GradientSaliency(m)
        x   = torch.randint(0, V, (1, 4))
        s   = sal.integrated_gradients(x, n_steps=3)
        assert s.method == "integrated_gradients"


class TestHeadAblatorCount:
    def test_result_count(self):
        abl = HeadAblator(MODEL)
        x   = torch.randint(0, V, (1, 4))
        y   = torch.randint(0, V, (1, 4))
        res = abl.ablate_all(x, y)
        # 2 layers × 2 heads = 4 results
        assert len(res) == 4
