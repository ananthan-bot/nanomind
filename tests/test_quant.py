"""tests/test_quant.py — Tests for NanoMind quantization package."""
import pytest
import torch
import torch.nn as nn
from nanomind.quant import (
    QuantConfig, TensorQuantizer, compute_scale_zero,
    quantize, dequantize, quantize_dequantize,
    RTNQuantizer, GPTQQuantizer, HessianCollector,
    AWQQuantizer, ActivationScaleCollector,
    QATLinear, convert_to_qat, fake_quant_ste,
    ModelCalibrator, LayerStats,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(16, 32)
        self.l2 = nn.Linear(32, 8)
    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


# ── QuantConfig ───────────────────────────────────────────────────────────────

class TestQuantConfig:
    def test_defaults(self):
        cfg = QuantConfig()
        assert cfg.bits == 8

    def test_invalid_bits(self):
        with pytest.raises(AssertionError):
            QuantConfig(bits=3)

    def test_invalid_scheme(self):
        with pytest.raises(AssertionError):
            QuantConfig(scheme="bad")

    def test_n_levels_int8(self):
        assert QuantConfig(bits=8).n_levels == 256

    def test_n_levels_int4(self):
        assert QuantConfig(bits=4).n_levels == 16

    def test_q_min_symmetric(self):
        cfg = QuantConfig(bits=8, scheme="symmetric")
        assert cfg.q_min == -128

    def test_q_max_asymmetric(self):
        cfg = QuantConfig(bits=8, scheme="asymmetric")
        assert cfg.q_max == 255

    def test_compression_ratio_int4(self):
        assert QuantConfig(bits=4).compression_ratio() == 8.0

    def test_bytes_per_param(self):
        assert QuantConfig(bits=4).bytes_per_param == 0.5

    def test_to_dict_keys(self):
        d = QuantConfig().to_dict()
        for k in ("bits", "scheme", "n_levels", "compression"):
            assert k in d


# ── compute_scale_zero ────────────────────────────────────────────────────────

class TestScaleZero:
    def test_symmetric_zero_is_zero(self):
        cfg = QuantConfig(bits=8, scheme="symmetric")
        x   = torch.randn(16, 8)
        s, z = compute_scale_zero(x, cfg)
        assert z.item() == 0.0

    def test_scale_positive(self):
        cfg = QuantConfig(bits=8)
        x   = torch.randn(16)
        s, _ = compute_scale_zero(x, cfg)
        assert s.item() > 0

    def test_per_channel_scale_shape(self):
        cfg = QuantConfig(bits=8, granularity="per_channel")
        x   = torch.randn(8, 16)
        s, z = compute_scale_zero(x, cfg, dim=1)
        assert s.shape == (8, 1)


# ── quantize / dequantize ─────────────────────────────────────────────────────

class TestQuantDequant:
    def test_quantize_output_dtype(self):
        cfg = QuantConfig(bits=8)
        x   = torch.randn(16)
        s, z = compute_scale_zero(x, cfg)
        q    = quantize(x, s, z, cfg)
        assert q.dtype == torch.int32

    def test_values_in_range(self):
        cfg = QuantConfig(bits=8, scheme="symmetric")
        x   = torch.randn(64)
        s, z = compute_scale_zero(x, cfg)
        q    = quantize(x, s, z, cfg)
        assert (q >= cfg.q_min).all() and (q <= cfg.q_max).all()

    def test_dequantize_close_to_original(self):
        cfg = QuantConfig(bits=8)
        x   = torch.randn(64)
        s, z = compute_scale_zero(x, cfg)
        q    = quantize(x, s, z, cfg)
        xr   = dequantize(q, s, z)
        assert (x - xr).abs().mean() < 0.1   # coarse check

    def test_qdq_float_output(self):
        cfg = QuantConfig(bits=8)
        x   = torch.randn(16)
        s, z = compute_scale_zero(x, cfg)
        xq   = quantize_dequantize(x, s, z, cfg)
        assert xq.dtype in (torch.float32, torch.float64)


# ── TensorQuantizer ───────────────────────────────────────────────────────────

class TestTensorQuantizer:
    def test_calibrate_and_error(self):
        q   = TensorQuantizer(QuantConfig(bits=8))
        x   = torch.randn(16, 8)
        err = q.quantization_error(x)
        for k in ("mse", "mae", "max_err", "snr_db"):
            assert k in err

    def test_snr_int8_higher_than_int4(self):
        x    = torch.randn(64, 32)
        q8   = TensorQuantizer(QuantConfig(bits=8))
        q4   = TensorQuantizer(QuantConfig(bits=4))
        e8   = q8.quantization_error(x)
        e4   = q4.quantization_error(x)
        assert e8["snr_db"] > e4["snr_db"]

    def test_fake_quantize_shape(self):
        q  = TensorQuantizer(QuantConfig(bits=4))
        x  = torch.randn(8, 16)
        xq = q.fake_quantize(x)
        assert xq.shape == x.shape


# ── RTNQuantizer ──────────────────────────────────────────────────────────────

class TestRTNQuantizer:
    def test_quantize_returns_dict(self):
        m   = TinyModel()
        rtn = RTNQuantizer(m, QuantConfig(bits=8))
        res = rtn.quantize()
        assert "l1" in res

    def test_model_size_bytes_smaller_than_fp32(self):
        m   = TinyModel()
        rtn = RTNQuantizer(m, QuantConfig(bits=4))
        rtn.quantize()
        fp32_size = sum(p.numel() for p in m.parameters()) * 4
        assert rtn.model_size_bytes() < fp32_size

    def test_layer_stats_count(self):
        m   = TinyModel()
        rtn = RTNQuantizer(m, QuantConfig(bits=8))
        rtn.quantize()
        stats = rtn.layer_stats()
        assert len(stats) == 2   # l1 and l2

    def test_layer_stats_keys(self):
        m   = TinyModel()
        rtn = RTNQuantizer(m, QuantConfig(bits=4))
        rtn.quantize()
        for s in rtn.layer_stats():
            assert "name" in s and "bits" in s and "params" in s


# ── GPTQQuantizer ─────────────────────────────────────────────────────────────

class TestGPTQQuantizer:
    def test_output_shape(self):
        W    = torch.randn(8, 16)
        H    = torch.eye(16)
        cfg  = QuantConfig(bits=4)
        gptq = GPTQQuantizer(W, H, cfg)
        W_q  = gptq.quantize()
        assert W_q.shape == W.shape

    def test_error_set_after_quantize(self):
        W    = torch.randn(8, 16)
        H    = torch.eye(16)
        gptq = GPTQQuantizer(W, H, QuantConfig(bits=4))
        gptq.quantize()
        assert isinstance(gptq.error, float)

    def test_hessian_collector_shape(self):
        m   = TinyModel()
        col = HessianCollector(m.l1)
        col.enable()
        m(torch.randn(4, 16))
        col.disable()
        H = col.hessian()
        assert H.shape == (16, 16)


# ── AWQQuantizer ──────────────────────────────────────────────────────────────

class TestAWQQuantizer:
    def test_output_shape(self):
        W     = torch.randn(8, 16)
        s     = torch.rand(16) + 0.5
        awq   = AWQQuantizer(W, s, QuantConfig(bits=4))
        W_q, scales = awq.quantize()
        assert W_q.shape == W.shape
        assert scales.shape == (16,)

    def test_error_dict_keys(self):
        W   = torch.randn(8, 16)
        s   = torch.rand(16)
        awq = AWQQuantizer(W, s, QuantConfig(bits=4))
        d   = awq.error()
        for k in ("mse", "max", "scale_mean"):
            assert k in d

    def test_activation_scale_collector_shape(self):
        m   = TinyModel()
        col = ActivationScaleCollector(m.l1)
        col.enable()
        m(torch.randn(4, 16))
        col.disable()
        s = col.scales()
        assert s.shape == (16,)


# ── QATLinear ─────────────────────────────────────────────────────────────────

class TestQATLinear:
    def test_forward_shape(self):
        qat = QATLinear(16, 32, cfg=QuantConfig(bits=4))
        x   = torch.randn(4, 16)
        assert qat(x).shape == (4, 32)

    def test_gradient_flows(self):
        qat  = QATLinear(16, 32, cfg=QuantConfig(bits=4))
        x    = torch.randn(4, 16)
        loss = qat(x).sum()
        loss.backward()
        assert qat.linear.weight.grad is not None

    def test_convert_to_qat(self):
        m   = TinyModel()
        cfg = QuantConfig(bits=4)
        m_q = convert_to_qat(m, cfg, in_place=False)
        n   = sum(1 for mod in m_q.modules() if isinstance(mod, QATLinear))
        assert n == 2

    def test_ste_gradient_passes_through(self):
        x = torch.randn(4, requires_grad=True)
        s = torch.tensor(0.1)
        z = torch.tensor(0.0)
        cfg = QuantConfig(bits=8, scheme="symmetric")
        from nanomind.quant import fake_quant_ste
        y = fake_quant_ste(x, s, z, cfg)
        y.sum().backward()
        assert x.grad is not None
        # STE: grad should pass through approximately
        assert x.grad.abs().sum() > 0


# ── ModelCalibrator ───────────────────────────────────────────────────────────

class TestModelCalibrator:
    def test_run_returns_stats(self):
        m   = TinyModel()
        cal = ModelCalibrator(m, QuantConfig(bits=4))
        stats = cal.run()
        assert len(stats) == 2   # 2 Linear layers

    def test_report_keys(self):
        m   = TinyModel()
        cal = ModelCalibrator(m, QuantConfig(bits=4))
        cal.run()
        r = cal.report()
        for k in ("n_layers", "mean_snr_db", "worst_layer", "bits"):
            assert k in r

    def test_sensitivity_analysis_ordered(self):
        m   = TinyModel()
        cal = ModelCalibrator(m, QuantConfig(bits=4))
        cal.run()
        sens = cal.sensitivity_analysis()
        snrs = [s["weight_snr"] for s in sens]
        assert snrs == sorted(snrs)

    def test_mixed_precision_returns_dict(self):
        m   = TinyModel()
        cal = ModelCalibrator(m, QuantConfig(bits=4))
        cal.run()
        mp = cal.mixed_precision_suggestion()
        assert isinstance(mp, dict)
        assert set(mp.values()) <= {4, 8}


class TestINT2:
    def test_int2_levels(self):
        cfg = QuantConfig(bits=2, scheme="symmetric")
        assert cfg.n_levels == 4

    def test_int2_quantize_in_range(self):
        cfg = QuantConfig(bits=2, scheme="symmetric")
        x   = torch.randn(32)
        s, z = compute_scale_zero(x, cfg)
        q    = quantize(x, s, z, cfg)
        assert (q >= cfg.q_min).all() and (q <= cfg.q_max).all()


class TestPerGroupQuantizer:
    def test_per_group_fake_quant_shape(self):
        cfg = QuantConfig(bits=4, granularity="per_group", group_size=8)
        q   = TensorQuantizer(cfg)
        x   = torch.randn(16, 32)
        xq  = q.fake_quantize(x)
        assert xq.shape == x.shape


class TestRTNApply:
    def test_apply_changes_weights(self):
        m    = TinyModel()
        w0   = m.l1.weight.data.clone()
        rtn  = RTNQuantizer(m, QuantConfig(bits=4))
        rtn.quantize()
        rtn.apply()
        w1   = m.l1.weight.data
        # Weights should have changed (quantization error)
        assert not torch.allclose(w0, w1)

    def test_apply_keeps_shape(self):
        m    = TinyModel()
        sh   = m.l1.weight.shape
        rtn  = RTNQuantizer(m, QuantConfig(bits=8))
        rtn.quantize()
        rtn.apply()
        assert m.l1.weight.shape == sh


class TestGPTQvsRTN:
    def test_gptq_error_recorded(self):
        W     = torch.randn(8, 16)
        H     = torch.eye(16)
        gptq  = GPTQQuantizer(W, H, QuantConfig(bits=4))
        gptq.quantize()
        # After quantization, error should be a small positive float
        assert gptq.error >= 0.0


class TestAWQAlpha:
    def test_alpha_zero_ignores_activations(self):
        W  = torch.randn(8, 16)
        s1 = torch.rand(16) + 0.5
        s2 = torch.rand(16) * 5 + 0.5   # very different activations
        a1 = AWQQuantizer(W, s1, QuantConfig(bits=4), alpha=0.0)
        a2 = AWQQuantizer(W, s2, QuantConfig(bits=4), alpha=0.0)
        # alpha=0: activation doesn't matter → same scale
        W1, _ = a1.quantize()
        W2, _ = a2.quantize()
        assert torch.allclose(W1, W2, atol=1e-4)


class TestCalibratorBits:
    def test_int8_better_snr_than_int4(self):
        m    = TinyModel()
        c8   = ModelCalibrator(m, QuantConfig(bits=8))
        c4   = ModelCalibrator(m, QuantConfig(bits=4))
        c8.run(); c4.run()
        r8   = c8.report(); r4 = c4.report()
        assert r8["mean_snr_db"] > r4["mean_snr_db"]
