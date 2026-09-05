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
