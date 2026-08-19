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


# ── Optimizer state ───────────────────────────────────────────────────────────

class TestOptimizerState:
    def test_optimizer_state_preserved(self, tmp_path):
        model = make_model()
        opt   = make_optimizer(model)

        # Run a step so optimizer has non-default state
        idx    = torch.randint(0, 32, (2, 8))
        tgt    = torch.randint(0, 32, (2, 8))
        _, loss = model(idx, tgt)
        loss.backward()
        opt.step(); opt.zero_grad()

        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, optimizer=opt, step=1)

        model2 = make_model()
        opt2   = make_optimizer(model2)
        load_checkpoint(path, model2, optimizer=opt2)

        # State dict keys should match
        assert opt.state_dict().keys() == opt2.state_dict().keys()

    def test_no_optimizer_state_when_not_saved(self, tmp_path):
        model = make_model()
        path  = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, optimizer=None, step=1)

        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert "optimizer_state" not in payload


# ── CheckpointManager ─────────────────────────────────────────────────────────

class TestCheckpointManager:
    def test_save_creates_file(self, tmp_path):
        cfg = CheckpointConfig(out_dir=str(tmp_path), save_best=False, keep_last_n=0)
        mgr = CheckpointManager(cfg)
        model = make_model()
        path = mgr.save(model, step=100, val_loss=2.0)
        assert path.exists()

    def test_best_ckpt_created_on_improvement(self, tmp_path):
        cfg = CheckpointConfig(out_dir=str(tmp_path), save_best=True, keep_last_n=0)
        mgr = CheckpointManager(cfg)
        model = make_model()
        mgr.save(model, step=100, val_loss=2.0)
        assert (tmp_path / "best.pt").exists()

    def test_best_ckpt_not_updated_on_no_improvement(self, tmp_path):
        cfg = CheckpointConfig(out_dir=str(tmp_path), save_best=True, keep_last_n=0)
        mgr = CheckpointManager(cfg)
        model = make_model()
        mgr.save(model, step=100, val_loss=2.0)
        import os
        mtime1 = os.path.getmtime(tmp_path / "best.pt")
        import time; time.sleep(0.05)
        mgr.save(model, step=200, val_loss=3.0)  # worse -> no update
        mtime2 = os.path.getmtime(tmp_path / "best.pt")
        assert mtime1 == mtime2

    def test_retention_policy(self, tmp_path):
        cfg = CheckpointConfig(out_dir=str(tmp_path), save_best=False, keep_last_n=2)
        mgr = CheckpointManager(cfg)
        model = make_model()
        for step in [100, 200, 300]:
            mgr.save(model, step=step, val_loss=1.0)
        # Only 2 most recent should remain
        pts = list(tmp_path.glob("step_*.pt"))
        assert len(pts) == 2


# ── list_checkpoints ──────────────────────────────────────────────────────────

class TestListCheckpoints:
    def test_empty_dir(self, tmp_path):
        cfg = CheckpointConfig(out_dir=str(tmp_path))
        mgr = CheckpointManager(cfg)
        assert mgr.list_checkpoints() == []

    def test_lists_all_checkpoints(self, tmp_path):
        cfg = CheckpointConfig(out_dir=str(tmp_path), keep_last_n=0)
        mgr = CheckpointManager(cfg)
        model = make_model()
        for step in [100, 200]:
            mgr.save(model, step=step, val_loss=1.0)
        ckpts = mgr.list_checkpoints()
        assert len(ckpts) == 2

    def test_metadata_in_list(self, tmp_path):
        cfg = CheckpointConfig(out_dir=str(tmp_path), keep_last_n=0)
        mgr = CheckpointManager(cfg)
        model = make_model()
        mgr.save(model, step=100, val_loss=1.23)
        ckpts = mgr.list_checkpoints()
        assert ckpts[0]["step"] == 100
        assert abs(ckpts[0]["val_loss"] - 1.23) < 1e-5


# ── Inference checkpoint ──────────────────────────────────────────────────────

class TestInferenceCheckpoint:
    def test_save_and_load_inference(self, tmp_path):
        model  = make_model()
        path   = tmp_path / "inference.pt"
        save_for_inference(path, model, model_config=CFG.to_dict(), step=500)
        assert path.exists()

        model2 = make_model()
        for p in model2.parameters():
            torch.nn.init.normal_(p)
        info = load_inference_checkpoint(path, model2)
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
        assert info["step"] == 500

    def test_inference_ckpt_has_no_optimizer(self, tmp_path):
        model = make_model()
        path  = tmp_path / "inference.pt"
        save_for_inference(path, model)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert "optimizer_state" not in payload


# ── auto_resume ───────────────────────────────────────────────────────────────

class TestAutoResume:
    def test_no_checkpoint_returns_zero(self, tmp_path):
        model = make_model()
        step, meta = auto_resume(str(tmp_path), model)
        assert step == 0
        assert meta is None

    def test_resumes_from_latest(self, tmp_path):
        model = make_model()
        save_checkpoint(tmp_path / "step_0000100.pt", model, step=100)
        save_checkpoint(tmp_path / "step_0000200.pt", model, step=200)
        model2 = make_model()
        step, meta = auto_resume(str(tmp_path), model2)
        assert step == 201   # step + 1
        assert meta["step"] == 200
