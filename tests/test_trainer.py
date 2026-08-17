"""
tests/test_trainer.py — Tests for the NanoMind Trainer.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nanomind.model import NanoMind, ModelConfig
from nanomind.trainer import Trainer, TrainConfig

# ── Fixtures ──────────────────────────────────────────────────────────────────

VOCAB  = 32
BLOCK  = 8
D      = 32
N_HEAD = 2
B      = 4


def make_model() -> NanoMind:
    torch.manual_seed(0)
    return NanoMind(ModelConfig(
        vocab_size=VOCAB, block_size=BLOCK,
        d_model=D, n_layers=2, n_heads=N_HEAD, dropout=0.0
    ))


def make_loaders(n: int = 64):
    tokens = torch.randint(0, VOCAB, (n + BLOCK,))
    xs = torch.stack([tokens[i:i+BLOCK]   for i in range(n)])
    ys = torch.stack([tokens[i+1:i+BLOCK+1] for i in range(n)])
    ds = TensorDataset(xs, ys)
    loader = DataLoader(ds, batch_size=B, shuffle=True, drop_last=True)
    return loader, loader  # use same for train/val in tests


def make_trainer(model=None, cfg=None) -> Trainer:
    m   = model or make_model()
    c   = cfg or TrainConfig(max_iters=10, eval_interval=5, log_interval=5,
                              grad_clip=1.0, use_amp=False)
    device = torch.device("cpu")
    tl, vl = make_loaders()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    return Trainer(m, opt, tl, vl, c, device)


# ── train_step ────────────────────────────────────────────────────────────────

class TestTrainStep:
    def test_returns_float(self):
        t = make_trainer()
        x, y = next(iter(t.train_loader))
        loss = t.train_step(x, y)
        assert isinstance(loss, float)

    def test_loss_is_positive(self):
        t = make_trainer()
        x, y = next(iter(t.train_loader))
        loss = t.train_step(x, y)
        assert loss > 0

    def test_multiple_steps_accumulate_grads(self):
        t = make_trainer()
        x, y = next(iter(t.train_loader))
        t.train_step(x, y)
        # At least some parameters should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in t.model.parameters()
        )
        assert has_grad


# ── Full training loop ────────────────────────────────────────────────────────

class TestTrainLoop:
    def test_loss_decreases(self):
        model = make_model()
        cfg   = TrainConfig(
            max_iters=50, eval_interval=25, log_interval=10,
            grad_clip=1.0, use_amp=False
        )
        device = torch.device("cpu")
        tl, vl = make_loaders(128)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        trainer = Trainer(model, opt, tl, vl, cfg, device)

        # Record initial loss
        x, y = next(iter(tl))
        initial_loss = trainer.eval_step(x, y)

        result = trainer.train()

        # Record final loss
        final_loss = trainer.eval_step(x, y)
        assert final_loss < initial_loss, (
            f"Expected loss to decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_train_returns_dict(self):
        t = make_trainer()
        result = t.train()
        assert "best_val" in result
        assert "final_train" in result


# ── estimate_loss ─────────────────────────────────────────────────────────────

class TestEstimateLoss:
    def test_returns_train_and_val(self):
        t = make_trainer()
        losses = t.estimate_loss()
        assert "train" in losses
        assert "val" in losses

    def test_losses_are_positive_floats(self):
        t = make_trainer()
        losses = t.estimate_loss()
        for split, v in losses.items():
            assert isinstance(v, float), f"{split} loss is not float"
            assert v > 0, f"{split} loss is not positive"

    def test_model_returns_to_train_mode(self):
        t = make_trainer()
        t.estimate_loss()
        assert t.model.training


# ── Gradient accumulation ─────────────────────────────────────────────────────

class TestGradAccum:
    def test_accum_produces_same_gradients(self):
        """2 steps with accum=2 should equal 1 step on double batch."""
        torch.manual_seed(42)
        model1 = make_model()
        torch.manual_seed(42)
        model2 = make_model()

        # Single step on batch of size 2B
        x2 = torch.randint(0, VOCAB, (B * 2, BLOCK))
        y2 = torch.randint(0, VOCAB, (B * 2, BLOCK))
        _, loss1 = model1(x2, y2)
        loss1.backward()

        # Two accumulated steps on batches of size B
        x1a = x2[:B]; y1a = y2[:B]
        x1b = x2[B:]; y1b = y2[B:]
        _, la = model2(x1a, y1a); (la / 2).backward()
        _, lb = model2(x1b, y1b); (lb / 2).backward()

        # Gradients should be approximately equal
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.grad is not None:
                assert torch.allclose(p1.grad, p2.grad, atol=1e-5), (
                    "Gradient mismatch between single-step and accumulated steps"
                )


# ── Early stopping ────────────────────────────────────────────────────────────

class TestEarlyStopping:
    def test_no_early_stop_when_patience_zero(self):
        t = make_trainer()
        assert not t._check_early_stop(999.0)   # Very high loss, but patience=0

    def test_early_stop_triggers(self):
        t = make_trainer(cfg=TrainConfig(
            max_iters=10, eval_interval=2, log_interval=5,
            early_stop_patience=2,
        ))
        t.best_val = 1.0
        t._check_early_stop(1.5)   # no improvement -> counter = 1
        assert t._check_early_stop(1.5)  # no improvement -> counter = 2 -> stop

    def test_early_stop_resets_on_improvement(self):
        t = make_trainer(cfg=TrainConfig(
            max_iters=10, eval_interval=2, log_interval=5,
            early_stop_patience=3,
        ))
        t.best_val = 2.0
        t._check_early_stop(1.5)   # improvement -> reset
        assert getattr(t, "_patience_counter", 0) == 0
