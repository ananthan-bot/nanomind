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
