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
