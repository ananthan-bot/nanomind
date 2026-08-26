"""
tests/test_lora.py — Tests for NanoMind LoRA fine-tuning.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from nanomind import NanoMind, ModelConfig
from nanomind.lora import (
    LoRAConfig,
    LoRALinear,
    LoRAModel,
    inject_lora,
    mark_only_lora_as_trainable,
    save_lora_checkpoint,
    load_lora_checkpoint,
    lora_parameter_stats,
    get_lora_state_dict,
    merge_all_lora,
    unmerge_all_lora,
)

IN_F, OUT_F, R = 64, 128, 8
B, T, D = 2, 8, 32
VOCAB = 32


def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=T, d_model=D,
                      n_layers=2, n_heads=4, dropout=0.0)
    return NanoMind(cfg)


# ── LoRALinear ────────────────────────────────────────────────────────────────

class TestLoRALinear:
    def test_output_shape(self):
        layer = LoRALinear(IN_F, OUT_F, r=R)
        x     = torch.randn(B, IN_F)
        out   = layer(x)
        assert out.shape == (B, OUT_F)

    def test_3d_input(self):
        layer = LoRALinear(IN_F, OUT_F, r=R)
        x     = torch.randn(B, T, IN_F)
        out   = layer(x)
        assert out.shape == (B, T, OUT_F)

    def test_lora_A_requires_grad(self):
        layer = LoRALinear(IN_F, OUT_F, r=R)
        assert layer.lora_A.requires_grad
        assert layer.lora_B.requires_grad

    def test_weight_frozen(self):
        layer = LoRALinear(IN_F, OUT_F, r=R)
        assert not layer.weight.requires_grad

    def test_lora_B_init_is_zero(self):
        layer = LoRALinear(IN_F, OUT_F, r=R)
        assert torch.all(layer.lora_B == 0)

    def test_from_linear(self):
        linear = nn.Linear(IN_F, OUT_F, bias=False)
        lora   = LoRALinear.from_linear(linear, r=R)
        assert torch.equal(lora.weight.data, linear.weight.data)

    def test_from_linear_with_bias(self):
        linear = nn.Linear(IN_F, OUT_F, bias=True)
        lora   = LoRALinear.from_linear(linear, r=R)
        assert lora.bias_param is not None
        assert torch.equal(lora.bias_param.data, linear.bias.data)


# ── Merge / Unmerge ───────────────────────────────────────────────────────────

class TestMergeUnmerge:
    def test_merge_changes_weight(self):
        layer  = LoRALinear(IN_F, OUT_F, r=R, alpha=16.0)
        w_orig = layer.weight.data.clone()
        # Train lora_A slightly
        layer.lora_A.data.fill_(0.01)
        layer.merge()
        assert not torch.equal(layer.weight.data, w_orig)

    def test_unmerge_restores_weight(self):
        layer  = LoRALinear(IN_F, OUT_F, r=R, alpha=16.0)
        w_orig = layer.weight.data.clone()
        layer.lora_A.data.fill_(0.01)
        layer.merge()
        layer.unmerge()
        assert torch.allclose(layer.weight.data, w_orig, atol=1e-6)

    def test_merged_output_equals_unmerged(self):
        layer = LoRALinear(IN_F, OUT_F, r=R, alpha=16.0)
        torch.manual_seed(1)
        layer.lora_A.data = torch.randn_like(layer.lora_A) * 0.01
        x       = torch.randn(B, IN_F)
        out_sep  = layer(x).detach().clone()
        layer.merge()
        out_merged = layer(x).detach().clone()
        assert torch.allclose(out_sep, out_merged, atol=1e-5)

    def test_double_merge_is_no_op(self):
        layer = LoRALinear(IN_F, OUT_F, r=R)
        w1    = layer.weight.data.clone()
        layer.merge()
        layer.merge()   # second merge should be no-op
        assert torch.allclose(layer.weight.data, w1, atol=1e-6)

    def test_merge_all_and_unmerge_all(self):
        model = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
        inject_lora(model, lora_cfg)
        merge_all_lora(model)
        unmerge_all_lora(model)   # should not raise
