"""
tests/test_checkpoint.py — Tests for the NanoMind checkpoint system.
"""

import pytest
import torch

from nanomind.model import NanoMind, ModelConfig
from nanomind.checkpoint import (
    CheckpointConfig,
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    load_for_inference,
    save_for_inference,
    load_inference_checkpoint,
    auto_resume,
    checkpoint_info,
)

CFG = ModelConfig(
    vocab_size=32, block_size=8,
    d_model=32, n_layers=2, n_heads=2, dropout=0.0
)


def make_model():
    torch.manual_seed(42)
    return NanoMind(CFG)


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


# ── save/load roundtrip ───────────────────────────────────────────────────────

class TestSaveLoadRoundtrip:
    def test_model_state_preserved(self, tmp_path):
        model = make_model()
        path  = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, step=100, val_loss=1.5)

        model2 = make_model()
        # Reinit model2 with different weights
        for p in model2.parameters():
            torch.nn.init.normal_(p)

        load_checkpoint(path, model2)
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2), "Model weights differ after load"

    def test_step_in_metadata(self, tmp_path):
        model = make_model()
        path  = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, step=42, val_loss=2.0)
        meta = load_checkpoint(path, make_model())
        assert meta["step"] == 42

    def test_companion_json_created(self, tmp_path):
        model = make_model()
        path  = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, step=1)
        assert (tmp_path / "ckpt.json").exists()
