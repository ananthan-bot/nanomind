"""
tests/test_quant.py — Tests for INT8 post-training quantization.
"""

import copy
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from nanomind import NanoMind, ModelConfig
from nanomind.quant import (
    QuantConfig,
    QuantizedLinear,
    DynamicQuantizedLinear,
    quantize_model,
    quantize_tensor,
    dequantize_tensor,
    quantization_stats,
    quantization_error,
    model_size_bytes,
    save_quantized_checkpoint,
    load_quantized_checkpoint,
)
from nanomind.quant.ops import (
    quantize_per_tensor, dequantize_per_tensor,
    quantize_per_channel, dequantize_per_channel,
)

IN_F, OUT_F = 64, 128
B, T, D = 2, 8, 64
VOCAB = 32


def tiny_model():
    torch.manual_seed(0)
    return NanoMind(ModelConfig(vocab_size=VOCAB, block_size=T, d_model=D,
                                n_layers=2, n_heads=4, dropout=0.0))


# ── quantize_tensor / dequantize_tensor ───────────────────────────────────────

class TestQuantizeOps:
    def test_per_tensor_output_dtype(self):
        x = torch.randn(IN_F, OUT_F)
        q, s = quantize_per_tensor(x)
        assert q.dtype == torch.int8

    def test_per_tensor_scale_scalar(self):
        x = torch.randn(IN_F, OUT_F)
        _, s = quantize_per_tensor(x)
        assert s.numel() == 1

    def test_per_tensor_roundtrip_close(self):
        x  = torch.randn(IN_F, OUT_F)
        q, s = quantize_per_tensor(x)
        xr = dequantize_per_tensor(q, s)
        # Roundtrip error should be small (< 1%)
        rel_err = (x - xr).abs().mean() / x.abs().mean()
        assert rel_err.item() < 0.05

    def test_per_channel_output_shape(self):
        x = torch.randn(OUT_F, IN_F)
        q, s = quantize_per_channel(x)
        assert q.shape == x.shape
        assert s.shape == (OUT_F,)

    def test_per_channel_roundtrip_close(self):
        x  = torch.randn(OUT_F, IN_F)
        q, s = quantize_per_channel(x)
        xr = dequantize_per_channel(q, s)
        rel_err = (x - xr).abs().mean() / x.abs().mean()
        assert rel_err.item() < 0.02  # per-channel more accurate

    def test_quantize_tensor_dispatch(self):
        x = torch.randn(OUT_F, IN_F)
        q_pt, _ = quantize_tensor(x, "per_tensor")
        q_pc, _ = quantize_tensor(x, "per_channel")
        assert q_pt.dtype == torch.int8
        assert q_pc.dtype == torch.int8

    def test_int8_range(self):
        x = torch.randn(OUT_F, IN_F)
        q, _ = quantize_per_channel(x)
        assert q.min().item() >= -128
        assert q.max().item() <= 127


# ── QuantizedLinear ───────────────────────────────────────────────────────────

class TestQuantizedLinear:
    def test_output_shape(self):
        ql = QuantizedLinear(IN_F, OUT_F)
        x  = torch.randn(B, IN_F)
        assert ql(x).shape == (B, OUT_F)

    def test_3d_input(self):
        ql = QuantizedLinear(IN_F, OUT_F)
        x  = torch.randn(B, T, IN_F)
        assert ql(x).shape == (B, T, OUT_F)

    def test_weight_stored_as_int8(self):
        ql = QuantizedLinear(IN_F, OUT_F)
        assert ql.weight_int8.dtype == torch.int8

    def test_from_linear_copies_weight(self):
        linear = nn.Linear(IN_F, OUT_F, bias=False)
        ql     = QuantizedLinear.from_linear(linear)
        # Dequantized weight should be close to original
        w_dq   = ql.weight
        rel_err = (linear.weight.data - w_dq).abs().mean() / linear.weight.data.abs().mean()
        assert rel_err.item() < 0.05

    def test_from_linear_with_bias(self):
        linear = nn.Linear(IN_F, OUT_F, bias=True)
        ql     = QuantizedLinear.from_linear(linear)
        assert ql.bias is not None
        assert torch.allclose(ql.bias.data, linear.bias.data)

    def test_scales_per_channel_shape(self):
        ql = QuantizedLinear(IN_F, OUT_F, granularity="per_channel")
        assert ql.scales.shape == (OUT_F,)

    def test_scales_per_tensor_shape(self):
        ql = QuantizedLinear(IN_F, OUT_F, granularity="per_tensor")
        assert ql.scales.numel() == 1


# ── quantize_model ────────────────────────────────────────────────────────────

class TestQuantizeModel:
    def test_linear_layers_replaced(self):
        model = tiny_model()
        qcfg  = QuantConfig(skip_modules=[])
        quantize_model(model, qcfg)
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                pytest.fail(f"Found un-quantized nn.Linear at {name}")

    def test_skip_modules_respected(self):
        model = tiny_model()
        qcfg  = QuantConfig(skip_modules=["lm_head"])
        quantize_model(model, qcfg)
        # lm_head should still be nn.Linear
        assert isinstance(model.lm_head, nn.Linear)

    def test_forward_still_works(self):
        model = tiny_model()
        quantize_model(model)
        idx = torch.randint(0, VOCAB, (B, T))
        logits, _ = model(idx)
        assert logits.shape == (B, T, VOCAB)

    def test_dynamic_mode_uses_dynamic_layer(self):
        model = tiny_model()
        qcfg  = QuantConfig(mode="dynamic", skip_modules=[])
        quantize_model(model, qcfg)
        n_dql = sum(1 for m in model.modules()
                    if isinstance(m, DynamicQuantizedLinear))
        assert n_dql > 0


# ── Size reduction ────────────────────────────────────────────────────────────

class TestSizeReduction:
    def test_quantized_smaller_than_original(self):
        original  = tiny_model()
        quantized = copy.deepcopy(original)
        quantize_model(quantized, QuantConfig(skip_modules=[]))
        assert model_size_bytes(quantized) < model_size_bytes(original)

    def test_compression_ratio_at_least_2x(self):
        original  = tiny_model()
        quantized = copy.deepcopy(original)
        quantize_model(quantized, QuantConfig(skip_modules=[]))
        stats = quantization_stats(original, quantized)
        # Linear weights 4x smaller; biases/embeddings remain float → overall ~2-3x
        assert stats["compression"] >= 1.5

    def test_quantization_error_low(self):
        original  = tiny_model()
        quantized = copy.deepcopy(original)
        quantize_model(quantized, QuantConfig(skip_modules=[]))
        err = quantization_error(original, quantized)
        assert err["mean_mse"] < 0.01

    def test_logit_mse_low(self):
        original  = tiny_model()
        quantized = copy.deepcopy(original)
        quantize_model(quantized, QuantConfig(skip_modules=[]))
        idx = torch.randint(0, VOCAB, (1, T))
        with torch.no_grad():
            l_fp, _ = original(idx)
            l_q, _  = quantized(idx)
        mse = ((l_fp - l_q) ** 2).mean().item()
        assert mse < 1.0   # logits should be reasonably close


# ── Quantized checkpoint ──────────────────────────────────────────────────────

class TestQuantizedCheckpoint:
    def test_save_creates_file(self, tmp_path):
        model = tiny_model()
        quantize_model(model)
        path = tmp_path / "quant.pt"
        save_quantized_checkpoint(model, path)
        assert path.exists()

    def test_quantized_smaller_than_fp32_checkpoint(self, tmp_path):
        import torch
        original  = tiny_model()
        quantized = copy.deepcopy(original)
        quantize_model(quantized, QuantConfig(skip_modules=[]))

        fp_path  = tmp_path / "fp32.pt"
        q_path   = tmp_path / "int8.pt"
        torch.save(original.state_dict(), fp_path)
        save_quantized_checkpoint(quantized, q_path)
        assert q_path.stat().st_size < fp_path.stat().st_size

    def test_roundtrip_preserves_weights(self, tmp_path):
        model    = tiny_model()
        quantize_model(model, QuantConfig(skip_modules=[]))

        path = tmp_path / "quant.pt"
        save_quantized_checkpoint(model, path)

        model2 = tiny_model()
        quantize_model(model2, QuantConfig(skip_modules=[]))
        load_quantized_checkpoint(model2, path)

        for p1, p2 in zip(model.buffers(), model2.buffers()):
            assert torch.equal(p1, p2)


# ── DynamicQuantizedLinear ────────────────────────────────────────────────────

class TestDynamicQuantizedLinear:
    def test_output_shape(self):
        dql = DynamicQuantizedLinear(IN_F, OUT_F)
        x   = torch.randn(B, IN_F)
        assert dql(x).shape == (B, OUT_F)

    def test_3d_input(self):
        dql = DynamicQuantizedLinear(IN_F, OUT_F)
        x   = torch.randn(B, T, IN_F)
        assert dql(x).shape == (B, T, OUT_F)

    def test_weight_int8(self):
        dql = DynamicQuantizedLinear(IN_F, OUT_F)
        assert dql.weight_int8.dtype == torch.int8

    def test_from_linear(self):
        linear = nn.Linear(IN_F, OUT_F)
        dql    = DynamicQuantizedLinear.from_linear(linear)
        out1   = linear(torch.zeros(B, IN_F))
        out2   = dql(torch.zeros(B, IN_F))
        # Zero input → both outputs should be the bias
        if linear.bias is not None:
            assert torch.allclose(out1, out2, atol=1e-4)


# ── QuantConfig ───────────────────────────────────────────────────────────────

class TestQuantConfig:
    def test_defaults(self):
        cfg = QuantConfig()
        assert cfg.mode == "weight_only"
        assert cfg.granularity == "per_channel"
        assert cfg.bits == 8

    def test_invalid_mode(self):
        with pytest.raises(AssertionError):
            QuantConfig(mode="int4")

    def test_invalid_granularity(self):
        with pytest.raises(AssertionError):
            QuantConfig(granularity="per_row")

    def test_invalid_bits(self):
        with pytest.raises(AssertionError):
            QuantConfig(bits=4)

    def test_skip_modules_list(self):
        cfg = QuantConfig(skip_modules=["lm_head", "embed"])
        assert "lm_head" in cfg.skip_modules
