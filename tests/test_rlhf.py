"""
tests/test_rlhf.py — Tests for RLHF components.
"""

import pytest
import torch
import torch.nn as nn

from nanomind import NanoMind, ModelConfig
from nanomind.rlhf import (
    RewardModelConfig, PPOConfig,
    RewardModel, PreferenceDataset,
    preference_loss, preference_accuracy, reward_stats,
    ValueHead, compute_gae,
    token_kl_divergence, approx_token_kl, AdaptiveKLController,
    PPORolloutBuffer, ppo_total_loss,
)
from nanomind.tokenizer.char import CharTokenizer

CORPUS = "abcdefghij " * 5
TOK    = CharTokenizer().build(CORPUS)
VOCAB  = TOK.vocab_size
B, T   = 2, 16
D      = 32

def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=T, d_model=D,
                      n_layers=2, n_heads=4, dropout=0.0)
    return NanoMind(cfg), cfg

def tiny_rm():
    m, cfg = tiny_model()
    return RewardModel(m, cfg, RewardModelConfig()), cfg


# ── RewardModelConfig ─────────────────────────────────────────────────────────

class TestRewardModelConfig:
    def test_defaults(self):
        cfg = RewardModelConfig()
        assert cfg.pooling == "last"

    def test_invalid_pooling(self):
        with pytest.raises(AssertionError):
            RewardModelConfig(pooling="max")

    def test_invalid_dropout(self):
        with pytest.raises(AssertionError):
            RewardModelConfig(dropout=1.5)


# ── RewardModel ───────────────────────────────────────────────────────────────

class TestRewardModel:
    def test_output_shape(self):
        rm, _ = tiny_rm()
        ids   = torch.randint(0, VOCAB, (B, T))
        r     = rm(ids)
        assert r.shape == (B,)

    def test_output_scalar_per_sequence(self):
        rm, _ = tiny_rm()
        ids   = torch.randint(0, VOCAB, (1, T))
        r     = rm(ids)
        assert r.ndim == 1

    def test_mean_pooling(self):
        m, cfg = tiny_model()
        rm     = RewardModel(m, cfg, RewardModelConfig(pooling="mean"))
        ids    = torch.randint(0, VOCAB, (B, T))
        r      = rm(ids)
        assert r.shape == (B,)

    def test_output_finite(self):
        rm, _ = tiny_rm()
        ids   = torch.randint(0, VOCAB, (B, T))
        assert rm(ids).isfinite().all()


# ── preference_loss ───────────────────────────────────────────────────────────

class TestPreferenceLoss:
    def test_loss_positive(self):
        r_w  = torch.tensor([1.0, 2.0])
        r_l  = torch.tensor([0.0, 1.5])
        loss = preference_loss(r_w, r_l)
        assert loss.item() > 0.0

    def test_perfect_ranking_low_loss(self):
        r_w  = torch.tensor([5.0] * 8)
        r_l  = torch.tensor([0.0] * 8)
        loss = preference_loss(r_w, r_l)
        assert loss.item() < 0.01

    def test_accuracy_all_correct(self):
        r_w = torch.tensor([1.0, 2.0, 3.0])
        r_l = torch.tensor([0.0, 1.0, 2.0])
        assert preference_accuracy(r_w, r_l) == 1.0

    def test_accuracy_all_wrong(self):
        r_w = torch.tensor([0.0, 0.0])
        r_l = torch.tensor([1.0, 1.0])
        assert preference_accuracy(r_w, r_l) == 0.0

    def test_reward_stats_keys(self):
        r_w = torch.randn(4)
        r_l = torch.randn(4)
        s   = reward_stats(r_w, r_l)
        assert all(k in s for k in ("mean_chosen","mean_rejected","mean_margin","accuracy"))


class TestPreferenceDataset:
    def test_len(self):
        pairs = [("a", "b", "c")] * 5
        ds    = PreferenceDataset(pairs, TOK, max_length=T)
        assert len(ds) == 5

    def test_item_shapes(self):
        pairs = [("ab", "cd", "ef")]
        ds    = PreferenceDataset(pairs, TOK, max_length=T)
        c, r  = ds[0]
        assert c.dtype == torch.long
        assert r.dtype == torch.long


# ── ValueHead + compute_gae ───────────────────────────────────────────────────

class TestValueHead:
    def test_output_shape(self):
        vh     = ValueHead(D)
        hidden = torch.randn(B, T, D)
        v      = vh(hidden)
        assert v.shape == (B, T)

    def test_gradient_flows(self):
        vh     = ValueHead(D)
        hidden = torch.randn(B, T, D, requires_grad=True)
        loss   = vh(hidden).sum()
        loss.backward()
        assert hidden.grad is not None


class TestComputeGAE:
    def test_output_shapes(self):
        T_     = 8
        r      = torch.zeros(T_)
        v      = torch.ones(T_)
        adv, ret = compute_gae(r, v, gamma=1.0, lam=0.95)
        assert adv.shape == (T_,)
        assert ret.shape == (T_,)

    def test_returns_equals_advantages_plus_values(self):
        T_     = 6
        r      = torch.rand(T_)
        v      = torch.rand(T_)
        adv, ret = compute_gae(r, v)
        assert torch.allclose(ret, adv + v, atol=1e-5)

    def test_zero_reward_negative_advantage(self):
        """With positive values and zero rewards, advantages should be negative."""
        T_     = 4
        r      = torch.zeros(T_)
        v      = torch.full((T_,), 2.0)
        adv, _ = compute_gae(r, v, gamma=1.0, lam=1.0, last_val=0.0)
        assert adv[-1].item() < 0.0


# ── KL divergence ─────────────────────────────────────────────────────────────

class TestKLDivergence:
    def test_same_policy_zero_kl(self):
        logits = torch.randn(B, T, VOCAB)
        kl     = token_kl_divergence(logits, logits)
        assert kl.abs().max().item() < 1e-5

    def test_kl_non_negative(self):
        logits_p = torch.randn(B, T, VOCAB)
        logits_q = torch.randn(B, T, VOCAB)
        kl       = token_kl_divergence(logits_p, logits_q)
        assert (kl >= 0).all()

    def test_approx_kl_shape(self):
        lp = torch.randn(B, T)
        lr = torch.randn(B, T)
        kl = approx_token_kl(lp, lr)
        assert kl.shape == (B, T)

    def test_adaptive_kl_increases_on_high_kl(self):
        ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=6.0)
        new  = ctrl.update(current_kl=10.0)
        assert new > 0.1

    def test_adaptive_kl_decreases_on_low_kl(self):
        ctrl = AdaptiveKLController(init_kl_coef=0.5, target_kl=6.0)
        new  = ctrl.update(current_kl=2.0)
        assert new < 0.5


# ── PPO losses ────────────────────────────────────────────────────────────────

class TestPPOLoss:
    def _make_tensors(self, n=16):
        lp  = torch.randn(n)
        olp = lp + 0.1 * torch.randn(n)
        adv = torch.randn(n)
        v   = torch.randn(n)
        ret = adv + v
        log = torch.randn(n, VOCAB)
        return lp, olp, adv, v, ret, log

    def test_total_loss_returns_dict(self):
        lp, olp, adv, v, ret, log = self._make_tensors()
        _, info = ppo_total_loss(lp, olp, adv, v, ret, log)
        for k in ("policy_loss","value_loss","entropy","total_loss","clip_fraction"):
            assert k in info

    def test_loss_scalar(self):
        lp, olp, adv, v, ret, log = self._make_tensors()
        total, _ = ppo_total_loss(lp, olp, adv, v, ret, log)
        assert total.ndim == 0

    def test_identical_policy_zero_clip(self):
        """If log_probs == old_log_probs, ratio=1, no clipping occurs."""
        lp  = torch.zeros(8)
        adv = torch.ones(8)
        v   = torch.zeros(8)
        ret = v
        _, info = ppo_total_loss(lp, lp, adv, v, ret, clip_ratio=0.2)
        assert info["clip_fraction"] == 0.0


# ── PPORolloutBuffer ──────────────────────────────────────────────────────────

class TestPPORolloutBuffer:
    def _make_rollout(self, T_gen=8):
        return (
            torch.randint(0, VOCAB, (T + T_gen,)),  # input_ids
            torch.randn(T_gen),                      # log_probs
            torch.randn(T_gen),                      # ref_log_probs
            torch.randn(T_gen),                      # values
            1.5,                                     # reward_score
        )

    def test_add_and_len(self):
        buf = PPORolloutBuffer()
        buf.add(*self._make_rollout())
        assert len(buf) == 1

    def test_finalize_returns_rollouts(self):
        buf = PPORolloutBuffer()
        for _ in range(3):
            buf.add(*self._make_rollout())
        rollouts = buf.finalize()
        assert len(rollouts) == 3

    def test_advantages_normalised(self):
        buf = PPORolloutBuffer()
        for _ in range(4):
            buf.add(*self._make_rollout())
        rollouts = buf.finalize()
        all_adv  = torch.cat([r.advantages for r in rollouts])
        assert all_adv.mean().abs().item() < 0.5   # close to zero mean
