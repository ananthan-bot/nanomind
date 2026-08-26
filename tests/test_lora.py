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


# ── inject_lora ───────────────────────────────────────────────────────────────

class TestInjectLoRA:
    def test_target_layers_replaced(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj"])
        inject_lora(model, lora_cfg)
        for name, mod in model.named_modules():
            if name.endswith("q_proj"):
                assert isinstance(mod, LoRALinear), f"{name} not replaced"

    def test_non_target_layers_unchanged(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj"])
        inject_lora(model, lora_cfg)
        for name, mod in model.named_modules():
            if name.endswith("v_proj"):
                assert isinstance(mod, nn.Linear), f"{name} should not be replaced"

    def test_mark_only_lora_trainable(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
        inject_lora(model, lora_cfg)
        mark_only_lora_as_trainable(model, bias="none")
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert p.requires_grad, f"{name} should be trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_model_still_runs_after_injection(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
        inject_lora(model, lora_cfg)
        idx = torch.randint(0, VOCAB, (B, T))
        logits, _ = model(idx)
        assert logits.shape == (B, T, VOCAB)


# ── LoRAModel ─────────────────────────────────────────────────────────────────

class TestLoRAModel:
    def test_forward_shape(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
        lm       = LoRAModel(model, lora_cfg)
        idx      = torch.randint(0, VOCAB, (B, T))
        logits, _ = lm(idx)
        assert logits.shape == (B, T, VOCAB)

    def test_lora_parameters_are_subset(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj"])
        lm       = LoRAModel(model, lora_cfg)
        lora_params = lm.lora_parameters()
        total_params = list(lm.parameters())
        assert len(lora_params) < len(total_params)

    def test_repr_contains_rank(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=8, target_modules=["q_proj"])
        lm       = LoRAModel(model, lora_cfg)
        assert "r=8" in repr(lm)

    def test_fewer_trainable_than_total(self):
        model = tiny_model()
        total_before = sum(p.numel() for p in model.parameters())
        lm    = LoRAModel(model, LoRAConfig(r=4, target_modules=["q_proj", "v_proj"]))
        stats = lora_parameter_stats(lm.model)
        assert stats["trainable"] < stats["total"]
        assert stats["lora_pct"] < 50.0


# ── LoRA checkpoint ───────────────────────────────────────────────────────────

class TestLoRACheckpoint:
    def test_save_creates_file(self, tmp_path):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj"])
        inject_lora(model, lora_cfg)
        path = tmp_path / "lora.pt"
        save_lora_checkpoint(model, path)
        assert path.exists()

    def test_checkpoint_smaller_than_full_model(self, tmp_path):
        import pickle, io
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
        inject_lora(model, lora_cfg)
        lora_path = tmp_path / "lora.pt"
        full_path = tmp_path / "full.pt"
        save_lora_checkpoint(model, lora_path)
        torch.save(model.state_dict(), full_path)
        assert lora_path.stat().st_size < full_path.stat().st_size

    def test_roundtrip_preserves_weights(self, tmp_path):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj"])
        inject_lora(model, lora_cfg)
        # Set A to non-zero
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.data.fill_(0.123)

        path = tmp_path / "lora.pt"
        save_lora_checkpoint(model, path)

        # Load into fresh model
        model2 = tiny_model()
        inject_lora(model2, lora_cfg)
        load_lora_checkpoint(model2, path)

        for m1, m2 in zip(model.modules(), model2.modules()):
            if isinstance(m1, LoRALinear):
                assert torch.equal(m1.lora_A.data, m2.lora_A.data)

    def test_lora_state_dict_only_lora_keys(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj"])
        inject_lora(model, lora_cfg)
        sd = get_lora_state_dict(model)
        assert all("lora_A" in k or "lora_B" in k for k in sd)


# ── Parameter stats ───────────────────────────────────────────────────────────

class TestParameterStats:
    def test_trainable_less_than_total(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
        inject_lora(model, lora_cfg)
        mark_only_lora_as_trainable(model)
        stats = lora_parameter_stats(model)
        assert stats["trainable"] < stats["total"]
        assert stats["frozen"] == stats["total"] - stats["trainable"]

    def test_lora_pct_in_range(self):
        model    = tiny_model()
        lora_cfg = LoRAConfig(r=2, target_modules=["q_proj"])
        inject_lora(model, lora_cfg)
        mark_only_lora_as_trainable(model)
        stats = lora_parameter_stats(model)
        assert 0.0 < stats["lora_pct"] < 100.0

    def test_higher_rank_more_trainable(self):
        model1 = tiny_model()
        model2 = tiny_model()
        inject_lora(model1, LoRAConfig(r=2, target_modules=["q_proj"]))
        inject_lora(model2, LoRAConfig(r=16, target_modules=["q_proj"]))
        mark_only_lora_as_trainable(model1)
        mark_only_lora_as_trainable(model2)
        s1 = lora_parameter_stats(model1)
        s2 = lora_parameter_stats(model2)
        assert s2["trainable"] > s1["trainable"]
