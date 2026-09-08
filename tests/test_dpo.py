"""
tests/test_dpo.py — Tests for DPO and Knowledge Distillation.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.dpo import DPOConfig, DPODataset, DPOTrainer, dpo_loss, compute_log_probs
from nanomind.distill import (
    DistillConfig, DistillTrainer,
    distillation_loss, soft_cross_entropy, feature_distillation_loss,
)

CORPUS = "abcdefghijklmnop " * 4
TOK    = CharTokenizer().build(CORPUS)
VOCAB  = TOK.vocab_size
B, T   = 2, 16
D      = 32

def tiny_model(d=D, layers=2):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=T, d_model=d,
                      n_layers=layers, n_heads=4, dropout=0.0)
    return NanoMind(cfg)


# ── DPOConfig ─────────────────────────────────────────────────────────────────

class TestDPOConfig:
    def test_defaults(self):
        cfg = DPOConfig()
        assert cfg.beta == 0.1
        assert cfg.loss_type == "sigmoid"

    def test_invalid_beta(self):
        with pytest.raises(AssertionError):
            DPOConfig(beta=0.0)

    def test_invalid_loss_type(self):
        with pytest.raises(AssertionError):
            DPOConfig(loss_type="ppo")

    def test_invalid_smoothing(self):
        with pytest.raises(AssertionError):
            DPOConfig(label_smoothing=0.6)


# ── dpo_loss ──────────────────────────────────────────────────────────────────

class TestDPOLoss:
    def test_loss_positive(self):
        lrc = torch.randn(B)
        lrr = torch.randn(B)
        loss, _ = dpo_loss(lrc, lrr)
        assert loss.item() > 0.0

    def test_returns_info_dict(self):
        lrc = torch.randn(4)
        lrr = torch.randn(4)
        _, info = dpo_loss(lrc, lrr)
        for k in ("loss","chosen_rewards","rejected_rewards","reward_margin","reward_accuracy"):
            assert k in info

    def test_perfect_separation_low_loss(self):
        """If chosen log-ratio >> rejected, loss should be near zero."""
        lrc = torch.tensor([5.0] * 8)
        lrr = torch.tensor([-5.0] * 8)
        loss, info = dpo_loss(lrc, lrr, beta=0.1)
        assert loss.item() < 0.01
        assert info["reward_accuracy"] == 1.0

    def test_ipo_loss_type(self):
        lrc = torch.randn(4)
        lrr = torch.randn(4)
        loss, _ = dpo_loss(lrc, lrr, loss_type="ipo")
        assert loss.item() >= 0.0

    def test_compute_log_probs_shape(self):
        logits = torch.randn(B, T, VOCAB)
        ids    = torch.randint(0, VOCAB, (B, T))
        lp     = compute_log_probs(logits, ids)
        assert lp.shape == (B,)


# ── DPODataset ────────────────────────────────────────────────────────────────

class TestDPODataset:
    def _pairs(self, n=4):
        return [("hello", " world", " moon")] * n

    def test_len(self):
        ds = DPODataset(self._pairs(6), TOK)
        assert len(ds) == 6

    def test_item_keys(self):
        ds   = DPODataset(self._pairs(), TOK)
        item = ds[0]
        assert "chosen_ids"   in item
        assert "rejected_ids" in item
        assert "prompt_len"   in item

    def test_collate_fn(self):
        ds     = DPODataset(self._pairs(4), TOK)
        batch  = [ds[i] for i in range(4)]
        result = DPODataset.collate_fn(batch)
        assert "chosen_ids"  in result
        assert "chosen_mask" in result
        assert result["chosen_ids"].shape[0] == 4

    def test_max_length_respected(self):
        cfg = DPOConfig(max_length=8)
        ds  = DPODataset([("a b c d e f g h", " i j", " k l")], TOK, cfg)
        item = ds[0]
        assert item["chosen_ids"].shape[0] <= 8


# ── DPOTrainer ────────────────────────────────────────────────────────────────

class TestDPOTrainer:
    def _make_trainer(self):
        model  = tiny_model()
        opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
        return DPOTrainer(model, opt, DPOConfig(beta=0.1)), model

    def test_reference_model_frozen(self):
        trainer, _ = self._make_trainer()
        for p in trainer.ref_model.parameters():
            assert not p.requires_grad

    def test_train_step_returns_dict(self):
        trainer, _ = self._make_trainer()
        pairs  = [("ab", "cd", "ef")] * 4
        ds     = DPODataset(pairs, TOK, DPOConfig(max_length=T))
        loader = DataLoader(ds, batch_size=4, collate_fn=DPODataset.collate_fn)
        batch  = next(iter(loader))
        result = trainer.train_step(batch)
        assert "loss" in result
        assert "reward_accuracy" in result

    def test_train_epoch_returns_steps(self):
        trainer, _ = self._make_trainer()
        pairs  = [("ab", "cd", "ef")] * 8
        ds     = DPODataset(pairs, TOK, DPOConfig(max_length=T))
        loader = DataLoader(ds, batch_size=4, collate_fn=DPODataset.collate_fn)
        m      = trainer.train_epoch(loader)
        assert m["steps"] == 2


# ── Distillation losses ───────────────────────────────────────────────────────

class TestDistillationLosses:
    def test_soft_cross_entropy_identical(self):
        """Same logits → minimal KL divergence."""
        logits = torch.randn(8, VOCAB)
        loss   = soft_cross_entropy(logits, logits, temperature=4.0)
        assert loss.item() < 1e-5

    def test_distillation_loss_keys(self):
        s = torch.randn(8, VOCAB)
        t = torch.randn(8, VOCAB)
        y = torch.randint(0, VOCAB, (8,))
        _, info = distillation_loss(s, t, y)
        assert all(k in info for k in ("loss","ce_loss","kd_loss"))

    def test_alpha_zero_pure_distillation(self):
        """alpha=0 → ce_loss weight is 0."""
        s = torch.randn(4, VOCAB)
        t = s.clone()   # same → kd_loss ≈ 0
        y = torch.randint(0, VOCAB, (4,))
        loss, info = distillation_loss(s, t, y, alpha=0.0)
        assert info["kd_loss"] < 1e-4

    def test_feature_distillation_loss(self):
        s = torch.randn(B, T, D)
        t = torch.randn(B, T, D)
        loss = feature_distillation_loss(s, t)
        assert loss.item() >= 0.0


# ── DistillTrainer ────────────────────────────────────────────────────────────

class TestDistillTrainer:
    def _make_trainer(self):
        teacher = tiny_model(d=64, layers=2)
        student = tiny_model(d=32, layers=2)
        opt     = torch.optim.Adam(student.parameters(), lr=1e-3)
        return DistillTrainer(teacher, student, opt, DistillConfig()), teacher, student

    def test_teacher_frozen(self):
        trainer, teacher, _ = self._make_trainer()
        for p in trainer.teacher.parameters():
            assert not p.requires_grad

    def test_compression_ratio_less_than_one(self):
        trainer, _, _ = self._make_trainer()
        assert 0 < trainer.compression_ratio() < 1.0

    def test_train_step_returns_loss(self):
        trainer, _, _ = self._make_trainer()
        x = torch.randint(0, VOCAB, (B, T))
        y = torch.randint(0, VOCAB, (B, T))
        m = trainer.train_step(x, y)
        assert "loss" in m and m["loss"] > 0.0
