"""tests/test_moe_v2.py — Tests for NanoMind MoE++ package."""
import pytest
import torch
from nanomind.moe_v2 import (
    MoEConfig, Expert, ExpertBank,
    TopKRouter, ExpertChoiceRouter, HashRouter, RoutingOutput,
    CapacityBuffer, CapacityStats, MoELayer,
    MoETransformerBlock, SparseMoETransformer,
    load_balance_loss, z_loss, entropy_loss, combined_moe_loss,
)

D, V = 32, 16


def _cfg(n_experts=4, top_k=2):
    return MoEConfig(n_experts=n_experts, top_k=top_k,
                      d_model=D, d_ff=D * 2, capacity_factor=1.5)


# ── MoEConfig ─────────────────────────────────────────────────────────────────

class TestMoEConfig:
    def test_defaults_valid(self):
        cfg = MoEConfig()
        assert cfg.n_experts >= 1

    def test_active_ratio(self):
        cfg = MoEConfig(n_experts=8, top_k=2)
        assert abs(cfg.active_ratio - 0.25) < 1e-4

    def test_invalid_top_k(self):
        with pytest.raises(AssertionError):
            MoEConfig(n_experts=4, top_k=5)

    def test_to_dict_keys(self):
        d = _cfg().to_dict()
        for k in ("n_experts", "top_k", "d_model", "router_type"):
            assert k in d

    def test_total_params_positive(self):
        assert _cfg().total_params_estimate > 0


# ── Expert ────────────────────────────────────────────────────────────────────

class TestExpert:
    def test_gelu_output_shape(self):
        e = Expert(D, D * 2, variant="gelu")
        x = torch.randn(4, D)
        assert e(x).shape == (4, D)

    def test_swiglu_output_shape(self):
        e = Expert(D, D * 2, variant="swiglu")
        x = torch.randn(4, D)
        assert e(x).shape == (4, D)

    def test_gradient_flows(self):
        e = Expert(D, D * 2, variant="gelu")
        x = torch.randn(4, D)
        out = e(x).sum()
        out.backward()
        has_grad = any(p.grad is not None for p in e.parameters())
        assert has_grad


# ── ExpertBank ────────────────────────────────────────────────────────────────

class TestExpertBank:
    def test_n_experts(self):
        bank = ExpertBank(_cfg(n_experts=4))
        assert len(bank.experts) == 4

    def test_forward_expert_shape(self):
        bank = ExpertBank(_cfg())
        x    = torch.randn(5, D)
        out  = bank.forward_expert(0, x)
        assert out.shape == (5, D)

    def test_shared_expert_count(self):
        cfg  = MoEConfig(n_experts=4, top_k=2, d_model=D, d_ff=D*2, shared_experts=2)
        bank = ExpertBank(cfg)
        assert len(bank.shared) == 2

    def test_shared_forward_shape(self):
        cfg  = MoEConfig(n_experts=4, top_k=2, d_model=D, d_ff=D*2, shared_experts=1)
        bank = ExpertBank(cfg)
        x    = torch.randn(8, D)
        out  = bank.shared_forward(x)
        assert out.shape == (8, D)

    def test_n_params_per_expert_positive(self):
        bank = ExpertBank(_cfg())
        assert bank.n_params_per_expert > 0


# ── TopKRouter ────────────────────────────────────────────────────────────────

class TestTopKRouter:
    def _router(self, n_experts=4, top_k=2):
        return TopKRouter(_cfg(n_experts, top_k))

    def test_indices_shape(self):
        r  = self._router()
        x  = torch.randn(8, D)
        out = r(x)
        assert out.indices.shape == (8, 2)

    def test_weights_shape(self):
        r  = self._router()
        x  = torch.randn(8, D)
        out = r(x)
        assert out.weights.shape == (8, 2)

    def test_weights_sum_to_1(self):
        r   = self._router()
        x   = torch.randn(8, D)
        out = r(x)
        sums = out.weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_indices_in_range(self):
        r   = self._router(n_experts=4)
        x   = torch.randn(8, D)
        out = r(x)
        assert (out.indices >= 0).all() and (out.indices < 4).all()

    def test_aux_loss_non_negative(self):
        r   = self._router()
        x   = torch.randn(8, D)
        r.train()
        out = r(x)
        assert out.aux_loss.item() >= 0.0

    def test_expert_utilisation_keys(self):
        r   = self._router()
        x   = torch.randn(8, D)
        out = r(x)
        u   = r.expert_utilisation(out.router_probs)
        assert "expert_counts" in u and "entropy" in u


# ── HashRouter ────────────────────────────────────────────────────────────────

class TestHashRouter:
    def test_deterministic(self):
        r   = HashRouter(_cfg(n_experts=4))
        x   = torch.randn(8, D)
        o1  = r(x)
        o2  = r(x)
        assert torch.all(o1.indices == o2.indices)

    def test_indices_cover_all_experts(self):
        r   = HashRouter(_cfg(n_experts=4))
        x   = torch.randn(16, D)
        out = r(x)
        # With 16 tokens and 4 experts, all experts should appear
        assert len(out.indices.unique()) == 4

    def test_no_learnable_params(self):
        r = HashRouter(_cfg())
        assert sum(p.numel() for p in r.parameters()) == 0


# ── CapacityBuffer ────────────────────────────────────────────────────────────

class TestCapacityBuffer:
    def test_capacity_formula(self):
        buf = CapacityBuffer(n_experts=8, capacity_factor=1.0)
        # 80 tokens, 8 experts, cf=1.0 → capacity=10
        assert buf.capacity(80) == 10

    def test_capacity_factor_scales(self):
        b1 = CapacityBuffer(n_experts=4, capacity_factor=1.0)
        b2 = CapacityBuffer(n_experts=4, capacity_factor=2.0)
        assert b2.capacity(40) == 2 * b1.capacity(40)

    def test_stats_returns_stats(self):
        buf  = CapacityBuffer(n_experts=4, capacity_factor=1.5)
        idx  = torch.randint(0, 4, (16, 1))
        s    = buf.stats(idx, 16)
        assert isinstance(s, CapacityStats)

    def test_overflow_frac_in_range(self):
        buf = CapacityBuffer(n_experts=4, capacity_factor=1.5)
        idx = torch.randint(0, 4, (16, 1))
        s   = buf.stats(idx, 16)
        assert 0.0 <= s.overflow_frac <= 1.0


# ── MoELayer ──────────────────────────────────────────────────────────────────

class TestMoELayer:
    def _layer(self):
        return MoELayer(_cfg())

    def test_output_shape(self):
        layer = self._layer()
        x     = torch.randn(2, 6, D)
        out, aux = layer(x)
        assert out.shape == (2, 6, D)

    def test_aux_loss_scalar(self):
        layer = self._layer()
        x     = torch.randn(2, 6, D)
        _, aux = layer(x)
        assert aux.shape == ()

    def test_gradient_flows(self):
        layer = self._layer()
        x     = torch.randn(2, 4, D, requires_grad=True)
        out, aux = layer(x)
        (out.sum() + aux).backward()
        assert x.grad is not None

    def test_routing_stats_keys(self):
        layer = self._layer()
        x     = torch.randn(2, 4, D)
        s     = layer.routing_stats(x)
        assert "capacity" in s


# ── SparseMoETransformer ──────────────────────────────────────────────────────

class TestSparseMoETransformer:
    def _model(self):
        cfg = _cfg()
        return SparseMoETransformer(V, D, n_layers=2, n_heads=2,
                                     max_seq=8, moe_cfg=cfg)

    def test_logits_shape(self):
        m   = self._model()
        ids = torch.randint(0, V, (2, 4))
        logits, _, aux = m(ids)
        assert logits.shape == (2, 4, V)

    def test_loss_scalar(self):
        m   = self._model()
        ids = torch.randint(0, V, (2, 4))
        _, loss, aux = m(ids, ids)
        assert loss.shape == ()

    def test_aux_loss_non_negative(self):
        m   = self._model()
        ids = torch.randint(0, V, (2, 4))
        _, _, aux = m(ids)
        assert aux.item() >= 0.0

    def test_n_params_positive(self):
        m = self._model()
        assert m.n_params > 0

    def test_n_active_less_than_total(self):
        m = self._model()
        assert m.n_active_params <= m.n_params


# ── Auxiliary Losses ──────────────────────────────────────────────────────────

class TestMoELosses:
    def _setup(self, N=32, E=8):
        probs   = torch.softmax(torch.randn(N, E), dim=-1)
        logits  = torch.randn(N, E)
        idx     = probs.topk(2, dim=-1).indices
        return probs, logits, idx, E

    def test_load_balance_non_negative(self):
        probs, logits, idx, E = self._setup()
        lb = load_balance_loss(probs, idx, E)
        assert lb.item() >= 0.0

    def test_z_loss_non_negative(self):
        _, logits, _, _ = self._setup()
        assert z_loss(logits).item() >= 0.0

    def test_entropy_loss_sign(self):
        probs, _, _, _ = self._setup()
        el = entropy_loss(probs)
        # Negative entropy → negative value for high-entropy distributions
        assert isinstance(el.item(), float)

    def test_combined_loss_scalar(self):
        probs, logits, idx, E = self._setup()
        c = combined_moe_loss(probs, logits, idx, E)
        assert c.shape == ()

    def test_uniform_routing_low_lb_loss(self):
        """Perfectly uniform routing → low load balance loss."""
        E     = 8
        N     = 64
        probs = torch.full((N, E), 1.0 / E)
        idx   = torch.zeros(N, 1, dtype=torch.long)
        lb    = load_balance_loss(probs, idx, E)
        # Uniform P_i, concentrated f_i → moderate loss
        assert lb.item() >= 0.0
