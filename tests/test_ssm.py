"""tests/test_ssm.py — Tests for NanoMind SSM package."""
import pytest
import torch
from nanomind.ssm import (
    SSMConfig, make_hippo_matrix, DiscretizedSSM,
    SelectiveSSM, MambaBlock, MambaLM,
    S4Layer, S4Block, S4Model,
    LinearAttnConfig, LinearAttention, RetentiveLayer,
    flops_comparison, parameter_count_comparison,
    compute_ssm_impulse_response, effective_memory_length,
    HybridConfig, HybridSSMTransformer,
)

V = 32


# ── HiPPO ─────────────────────────────────────────────────────────────────────

class TestHiPPO:
    def test_shape(self):
        A = make_hippo_matrix(8)
        assert A.shape == (8, 8)

    def test_lower_triangular(self):
        A = make_hippo_matrix(8)
        assert (A.triu(diagonal=1) == 0).all()

    def test_diagonal_positive(self):
        A = make_hippo_matrix(8)
        assert (A.diagonal() > 0).all()


# ── DiscretizedSSM ────────────────────────────────────────────────────────────

class TestDiscretizedSSM:
    def test_conv_output_shape(self):
        ssm = DiscretizedSSM(d_state=4)
        x   = torch.randn(2, 16, 1)
        y   = ssm.forward_conv(x)
        assert y.shape == x.shape

    def test_recurrent_output_shape(self):
        ssm = DiscretizedSSM(d_state=4)
        x   = torch.randn(2, 16, 1)
        h, y = ssm.forward_recurrent(x)
        assert y.shape == x.shape

    def test_conv_recurrent_close(self):
        ssm  = DiscretizedSSM(d_state=4)
        x    = torch.randn(1, 8, 1)
        y_c  = ssm.forward_conv(x)
        _, y_r = ssm.forward_recurrent(x)
        assert torch.allclose(y_c, y_r, atol=1e-4)

    def test_dt_positive(self):
        ssm = DiscretizedSSM(d_state=4)
        assert ssm.dt.item() > 0


# ── Mamba ──────────────────────────────────────────────────────────────────────

class TestMamba:
    def _cfg(self):
        return SSMConfig(d_model=16, d_state=4, d_conv=4, expand=2)

    def test_selective_ssm_shape(self):
        ssm = SelectiveSSM(d_inner=32, d_state=4)
        x   = torch.randn(2, 8, 32)
        y   = ssm(x)
        assert y.shape == x.shape

    def test_mamba_block_shape(self):
        block = MambaBlock(self._cfg())
        x     = torch.randn(2, 8, 16)
        y     = block(x)
        assert y.shape == x.shape

    def test_mamba_block_residual(self):
        block = MambaBlock(self._cfg())
        x     = torch.randn(2, 8, 16)
        y     = block(x)
        # Residual means output is not identical to input
        assert not torch.allclose(x, y)

    def test_mamba_lm_shape(self):
        cfg = self._cfg()
        lm  = MambaLM(V, cfg, n_layers=2)
        ids = torch.randint(0, V, (2, 8))
        y   = lm(ids)
        assert y.shape == (2, 8, V)

    def test_mamba_lm_params(self):
        cfg = self._cfg()
        lm  = MambaLM(V, cfg, n_layers=2)
        assert lm.n_parameters() > 0

    def test_weight_tying(self):
        cfg = self._cfg()
        lm  = MambaLM(V, cfg, n_layers=2)
        # Weight tying: embedding and lm_head share weights
        assert lm.lm_head.weight is lm.embedding.weight


# ── S4 ────────────────────────────────────────────────────────────────────────

class TestS4:
    def test_s4_layer_shape(self):
        layer = S4Layer(d_model=8, d_state=4)
        x     = torch.randn(2, 16, 8)
        y     = layer(x)
        assert y.shape == x.shape

    def test_s4_block_shape(self):
        block = S4Block(d_model=8, d_state=4, d_ff=32)
        x     = torch.randn(2, 16, 8)
        y     = block(x)
        assert y.shape == x.shape

    def test_s4_model_shape(self):
        model = S4Model(vocab_size=V, d_model=8, d_state=4, n_layers=2, d_ff=32)
        ids   = torch.randint(0, V, (2, 8))
        logits = model(ids)
        assert logits.shape == (2, 8, V)

    def test_s4_model_params(self):
        model = S4Model(vocab_size=V, d_model=8, d_state=4, n_layers=2, d_ff=32)
        assert model.n_parameters() > 0


# ── Linear Attention ──────────────────────────────────────────────────────────

class TestLinearAttn:
    def _la(self):
        return LinearAttention(LinearAttnConfig(d_model=16, n_heads=2, d_head=8))

    def test_output_shape(self):
        la = self._la()
        x  = torch.randn(2, 8, 16)
        y  = la(x)
        assert y.shape == x.shape

    def test_recurrent_step_shape(self):
        la  = self._la()
        x_0 = torch.randn(2, 16)
        S   = torch.zeros(2, 2, 8, 8)
        z   = torch.zeros(2, 2, 8)
        y, S1, z1 = la.recurrent_step(x_0, S, z)
        assert y.shape == (2, 16)

    def test_retentive_shape(self):
        ret = RetentiveLayer(d_model=16, n_heads=2)
        x   = torch.randn(2, 8, 16)
        y   = ret(x)
        assert y.shape == x.shape

    def test_retentive_gamma_range(self):
        ret = RetentiveLayer(d_model=16, n_heads=2)
        assert all(0 < g < 1 for g in ret.gamma.tolist())


# ── Hybrid ────────────────────────────────────────────────────────────────────

class TestHybrid:
    def _model(self):
        cfg = HybridConfig(d_model=16, n_heads=2, d_state=4, n_layers=4,
                            attn_every=2, vocab_size=V)
        return HybridSSMTransformer(cfg)

    def test_output_shape(self):
        m   = self._model()
        ids = torch.randint(0, V, (2, 8))
        y   = m(ids)
        assert y.shape == (2, 8, V)

    def test_layer_summary(self):
        m = self._model()
        s = m.layer_type_summary()
        assert s["total_layers"] == 4
        assert s["attn_layers"] == 2   # attn_every=2 → layers 2,4

    def test_params_positive(self):
        m = self._model()
        assert m.n_parameters() > 0


# ── Analysis ──────────────────────────────────────────────────────────────────

class TestAnalysis:
    def test_flops_comparison_keys(self):
        f = flops_comparison(2, 128, 64, 16)
        assert "transformer_GFLOPs" in f and "mamba_GFLOPs" in f

    def test_transformer_flops_gt_mamba(self):
        f = flops_comparison(2, 512, 64, 16)
        assert f["transformer_GFLOPs"] > f["mamba_GFLOPs"]

    def test_parameter_count_keys(self):
        p = parameter_count_comparison(64, 8, 128, 256, 4)
        assert "transformer_M" in p and "mamba_M" in p

    def test_impulse_response_shape(self):
        ssm = DiscretizedSSM(d_state=4)
        A_bar, B_bar = ssm._discretise()
        C = ssm.C[..., 0] + 1j * ssm.C[..., 1]
        K = compute_ssm_impulse_response(A_bar, B_bar.real, C.real, T=8)
        assert K.shape == (8,)

    def test_effective_memory_length(self):
        K = torch.tensor([1.0, 0.5, 0.2, 0.05, 0.01])
        eff = effective_memory_length(K, threshold=0.1)
        assert eff >= 1


class TestSSMKernel:
    def test_kernel_shape(self):
        ssm = DiscretizedSSM(d_state=4)
        K   = ssm._kernel(T=16)
        assert K.shape == (16,)

    def test_kernel_real_valued(self):
        ssm = DiscretizedSSM(d_state=4)
        K   = ssm._kernel(T=8)
        assert K.dtype in (torch.float32, torch.float64)
