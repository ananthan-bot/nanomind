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
