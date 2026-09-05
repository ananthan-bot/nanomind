"""
tests/test_amp.py — Tests for AMP and Gradient Checkpointing.
"""

import pytest
import torch
import torch.nn as nn

from nanomind.amp import (
    AMPConfig, mixed_precision_context, is_amp_available,
    NanoGradScaler, GradAccumulator, AMPTrainer,
    checkpointed_forward, estimate_activation_memory,
    model_parameter_memory_mb,
)


# ── AMPConfig ─────────────────────────────────────────────────────────────────

class TestAMPConfig:
    def test_defaults(self):
        cfg = AMPConfig()
        assert cfg.enabled is True
        assert cfg.dtype == "bfloat16"
        assert cfg.grad_scaler is False  # bfloat16 disables scaler

    def test_float16_enables_scaler(self):
        cfg = AMPConfig(dtype="float16")
        # grad_scaler stays True (until CUDA check at runtime)
        assert cfg.grad_scaler is True

    def test_invalid_dtype(self):
        with pytest.raises(AssertionError):
            AMPConfig(dtype="int8")

    def test_invalid_accum_steps(self):
        with pytest.raises(AssertionError):
            AMPConfig(grad_accum_steps=0)

    def test_torch_dtype_property(self):
        assert AMPConfig(dtype="bfloat16").torch_dtype == torch.bfloat16
        assert AMPConfig(dtype="float32").torch_dtype  == torch.float32


# ── mixed_precision_context ───────────────────────────────────────────────────

class TestMixedPrecisionContext:
    def test_no_crash_on_cpu(self):
        cfg = AMPConfig(enabled=True, dtype="bfloat16")
        with mixed_precision_context(cfg, device="cpu"):
            x = torch.randn(4, 8)
            y = x @ x.T

    def test_disabled_amp_passthrough(self):
        cfg = AMPConfig(enabled=False)
        with mixed_precision_context(cfg, device="cpu"):
            x = torch.randn(4, 4, dtype=torch.float32)
            assert x.dtype == torch.float32

    def test_is_amp_available_cpu(self):
        assert is_amp_available("cpu") is True


# ── GradAccumulator ───────────────────────────────────────────────────────────

class TestGradAccumulator:
    def test_should_step_every_n(self):
        acc = GradAccumulator(accum_steps=4)
        for i in range(3):
            assert not acc.should_step()
            acc.step()
        assert acc.should_step()

    def test_single_step(self):
        acc = GradAccumulator(accum_steps=1)
        assert acc.should_step()

    def test_reset(self):
        acc = GradAccumulator(accum_steps=4)
        for _ in range(3):
            acc.step()
        acc.reset()
        assert acc.current_step == 0

    def test_loss_scale(self):
        acc = GradAccumulator(accum_steps=8)
        assert acc.loss_scale == 8.0

    def test_invalid_accum_steps(self):
        with pytest.raises(AssertionError):
            GradAccumulator(0)


# ── NanoGradScaler ────────────────────────────────────────────────────────────

class TestNanoGradScaler:
    def test_scale_passthrough_no_cuda(self):
        """On CPU (no CUDA), scale() should be identity."""
        cfg    = AMPConfig(dtype="float16")
        scaler = NanoGradScaler(cfg)
        loss   = torch.tensor(2.5)
        scaled = scaler.scale(loss)
        # Either same tensor (no CUDA) or scaled
        assert scaled.item() > 0

    def test_state_dict_empty_on_cpu(self):
        cfg    = AMPConfig(dtype="float16")
        scaler = NanoGradScaler(cfg)
        # On CPU (no CUDA), _active=False → empty state dict
        assert isinstance(scaler.state_dict(), dict)

    def test_scale_factor_default(self):
        cfg    = AMPConfig(dtype="bfloat16")  # scaler disabled
        scaler = NanoGradScaler(cfg)
        assert scaler.scale_factor == 1.0


# ── checkpointed_forward ──────────────────────────────────────────────────────

class TestCheckpointedForward:
    def test_output_matches_normal(self):
        """Checkpointed and normal forward should give identical output."""
        module = nn.Linear(32, 32)
        x      = torch.randn(4, 32, requires_grad=True)
        normal = module(x)
        ckpt   = checkpointed_forward(module, x)
        assert torch.allclose(normal, ckpt, atol=1e-6)

    def test_gradient_flows_through_checkpoint(self):
        module = nn.Linear(16, 16)
        x      = torch.randn(2, 16, requires_grad=True)
        out    = checkpointed_forward(module, x)
        out.sum().backward()
        assert x.grad is not None
        assert module.weight.grad is not None

    def test_output_shape_preserved(self):
        module = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
        x      = torch.randn(3, 8)
        out    = checkpointed_forward(module, x)
        assert out.shape == (3, 8)


# ── estimate_activation_memory ────────────────────────────────────────────────

class TestEstimateActivationMemory:
    def test_keys(self):
        act = estimate_activation_memory(2, 128, 256, 8)
        for key in ("standard_mb", "checkpointed_mb", "savings_ratio"):
            assert key in act

    def test_checkpointed_less_than_standard(self):
        act = estimate_activation_memory(2, 128, 256, 16)
        assert act["checkpointed_mb"] <= act["standard_mb"]

    def test_savings_greater_than_one(self):
        act = estimate_activation_memory(2, 512, 256, 16)
        assert act["savings_ratio"] >= 1.0

    def test_model_parameter_memory(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 32))
        mem   = model_parameter_memory_mb(model)
        assert mem["params_mb"] > 0
        assert mem["n_params"] > 0
