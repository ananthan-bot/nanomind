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
