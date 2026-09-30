"""tests/test_alignment.py — Tests for NanoMind alignment package."""
import pytest
import torch
import torch.nn as nn
from nanomind.alignment import (
    PreferencePair, BinaryFeedback, PreferenceDataset,
    DPOConfig, dpo_loss, compute_log_probs, DPOTrainer,
    ConstitutionalAI, ConstitutionalConfig,
    KTOConfig, kto_loss, KTOTrainer,
    RewardModel,
    AlignmentEvaluator, AlignmentMetrics,
)

V = 16

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 8)
        self.rnn  = nn.GRU(8, 16, batch_first=True)
        self.head = nn.Linear(16, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None


# ── PreferenceDataset ─────────────────────────────────────────────────────────

class TestPreferenceDataset:
    def test_add_and_len(self):
        d = PreferenceDataset()
        d.add(PreferencePair("p", "c", "r"))
        assert len(d) == 1

    def test_filter_by_margin(self):
        d = PreferenceDataset([
            PreferencePair("p", "c", "r", score_chosen=0.9, score_rejected=0.1),
            PreferencePair("p", "c", "r", score_chosen=0.6, score_rejected=0.5),
        ])
        f = d.filter_by_margin(0.5)
        assert len(f) == 1

    def test_filter_by_source(self):
        d = PreferenceDataset([
            PreferencePair("p", "c", "r", source="human"),
            PreferencePair("p", "c", "r", source="ai"),
        ])
        assert len(d.filter_by_source("human")) == 1

    def test_stats_keys(self):
        d = PreferenceDataset([PreferencePair("p", "c", "r")])
        s = d.stats()
        assert "n_pairs" in s

    def test_to_binary(self):
        d = PreferenceDataset([PreferencePair("p", "c", "r")])
        b = d.to_binary()
        assert len(b) == 2
        assert any(x.is_good for x in b)
        assert any(not x.is_good for x in b)

    def test_flip(self):
        p = PreferencePair("q", "chosen", "rejected")
        f = p.flip()
        assert f.chosen == "rejected"
        assert f.rejected == "chosen"


# ── DPO ───────────────────────────────────────────────────────────────────────

class TestDPO:
    def _logps(self, B=4):
        pw = torch.tensor([-2.0] * B)
        pl = torch.tensor([-3.0] * B)
        rw = torch.tensor([-2.1] * B)
        rl = torch.tensor([-2.9] * B)
        return pw, pl, rw, rl

    def test_dpo_loss_scalar(self):
        pw, pl, rw, rl = self._logps()
        l, m = dpo_loss(pw, pl, rw, rl, DPOConfig(loss_type="dpo"))
        assert l.shape == ()

    def test_ipo_loss_scalar(self):
        pw, pl, rw, rl = self._logps()
        l, m = dpo_loss(pw, pl, rw, rl, DPOConfig(loss_type="ipo"))
        assert l.shape == ()

    def test_kto_pair_loss(self):
        pw, pl, rw, rl = self._logps()
        l, m = dpo_loss(pw, pl, rw, rl, DPOConfig(loss_type="kto_pair"))
        assert l.item() >= 0

    def test_metrics_keys(self):
        pw, pl, rw, rl = self._logps()
        _, m = dpo_loss(pw, pl, rw, rl, DPOConfig())
        for k in ("loss", "accuracy", "reward_chosen", "reward_rejected", "reward_margin"):
            assert k in m

    def test_accuracy_in_range(self):
        pw, pl, rw, rl = self._logps()
        _, m = dpo_loss(pw, pl, rw, rl, DPOConfig())
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_dpo_trainer_step(self):
        p   = TinyLM(); r = TinyLM()
        opt = torch.optim.AdamW(p.parameters())
        t   = DPOTrainer(p, r, opt, DPOConfig())
        c   = torch.randint(0, V, (2, 4))
        rej = torch.randint(0, V, (2, 4))
        m   = t.step(c, rej)
        assert "loss" in m

    def test_ref_model_frozen(self):
        p   = TinyLM(); r = TinyLM()
        opt = torch.optim.AdamW(p.parameters())
        DPOTrainer(p, r, opt)
        for param in r.parameters():
            assert not param.requires_grad


# ── Constitutional AI ──────────────────────────────────────────────────────────

class TestConstitutionalAI:
    def test_critique_returns_str(self):
        cai = ConstitutionalAI()
        c   = cai.critique("a response", "be helpful")
        assert isinstance(c, str) and len(c) > 0

    def test_revise_returns_str(self):
        cai = ConstitutionalAI()
        r   = cai.revise("bad response", "it is bad", "be helpful")
        assert isinstance(r, str)

    def test_critique_and_revise(self):
        cai = ConstitutionalAI()
        cr  = cai.critique_and_revise("some response")
        assert cr.original_response == "some response"
        assert cr.revised_response != ""
        assert cr.principle != ""

    def test_ai_feedback_returns_AB(self):
        cai = ConstitutionalAI()
        fb  = cai.ai_feedback("prompt", "response A", "response B")
        assert fb in ("A", "B")

    def test_build_preference_dataset(self):
        cai = ConstitutionalAI()
        ds  = cai.build_preference_dataset(["p"], ["a"], ["b"])
        assert len(ds) == 1
        assert ds[0].source == "ai"

    def test_generate_sft_data(self):
        cai  = ConstitutionalAI()
        data = cai.generate_sft_data(["q"], ["resp"])
        assert len(data) == 1
        assert "prompt" in data[0] and "response" in data[0]


# ── KTO ───────────────────────────────────────────────────────────────────────

class TestKTO:
    def test_kto_loss_scalar(self):
        lp = torch.tensor([-2.0, -3.0, -2.5, -3.5])
        rl = torch.tensor([-2.1, -2.9, -2.4, -3.3])
        kl = torch.tensor(0.1)
        is_d = torch.tensor([True, False, True, False])
        l, m = kto_loss(lp, rl, is_d, kl, KTOConfig())
        assert l.shape == ()

    def test_kto_metrics_keys(self):
        lp = torch.tensor([-2.0, -3.0])
        rl = torch.tensor([-2.1, -2.9])
        kl = torch.tensor(0.05)
        is_d = torch.tensor([True, False])
        _, m = kto_loss(lp, rl, is_d, kl, KTOConfig())
        for k in ("loss", "reward_desirable", "reward_undesirable", "kl_estimate"):
            assert k in m

    def test_kto_trainer_step(self):
        p   = TinyLM(); r = TinyLM()
        opt = torch.optim.AdamW(p.parameters())
        t   = KTOTrainer(p, r, opt, KTOConfig())
        ids = torch.randint(0, V, (4, 6))
        is_d = torch.tensor([True, False, True, False])
        m = t.step(ids, is_d)
        assert "loss" in m


# ── RewardModel ───────────────────────────────────────────────────────────────

class TestRewardModel:
    def test_forward_shape(self):
        rm = RewardModel(TinyLM(), d_model=V)
        ids = torch.randint(0, V, (4, 8))
        r   = rm(ids)
        assert r.shape == (4,)

    def test_preference_loss_keys(self):
        rm = RewardModel(TinyLM(), d_model=V)
        c  = torch.randint(0, V, (2, 6))
        r  = torch.randint(0, V, (2, 6))
        l, m = rm.preference_loss(c, r)
        for k in ("loss", "accuracy", "reward_margin"):
            assert k in m

    def test_accuracy_in_range(self):
        rm = RewardModel(TinyLM(), d_model=V)
        c  = torch.randint(0, V, (4, 6))
        r  = torch.randint(0, V, (4, 6))
        _, m = rm.preference_loss(c, r)
        assert 0.0 <= m["accuracy"] <= 1.0


# ── AlignmentEvaluator ────────────────────────────────────────────────────────

class TestAlignmentEvaluator:
    def test_evaluate_returns_metrics(self):
        rm   = RewardModel(TinyLM(), V)
        ev   = AlignmentEvaluator(rm, TinyLM())
        ids  = torch.randint(0, V, (4, 6))
        m    = ev.evaluate(TinyLM(), TinyLM(), ids)
        assert isinstance(m, AlignmentMetrics)

    def test_win_rate_in_range(self):
        rm   = RewardModel(TinyLM(), V)
        ev   = AlignmentEvaluator(rm, TinyLM())
        ids  = torch.randint(0, V, (4, 6))
        wr   = ev.win_rate(TinyLM(), TinyLM(), ids)
        assert 0.0 <= wr <= 1.0

    def test_reward_hacking_detection(self):
        m = AlignmentMetrics(win_rate=0.9, mean_reward=3.0, kl_div=3.0, reward_std=0.5)
        assert m.is_reward_hacking()

    def test_no_reward_hacking_normal(self):
        m = AlignmentMetrics(win_rate=0.6, mean_reward=0.5, kl_div=0.1, reward_std=0.2)
        assert not m.is_reward_hacking()

    def test_to_dict_keys(self):
        m = AlignmentMetrics(win_rate=0.7, mean_reward=1.0, kl_div=0.3, reward_std=0.2)
        d = m.to_dict()
        for k in ("win_rate", "mean_reward", "kl_div", "reward_std"):
            assert k in d


class TestDPOConfigValidation:
    def test_invalid_loss_type(self):
        import pytest
        with pytest.raises(AssertionError):
            DPOConfig(loss_type="bad")


class TestDPORewardOrdering:
    def test_good_pairs_have_positive_margin(self):
        # If chosen clearly better, margin should be positive
        pw = torch.tensor([-1.0] * 4)   # high chosen log-p
        pl = torch.tensor([-5.0] * 4)   # low rejected log-p
        rw = torch.tensor([-2.0] * 4)
        rl = torch.tensor([-4.0] * 4)
        _, m = dpo_loss(pw, pl, rw, rl, DPOConfig())
        assert m["reward_margin"] > 0


class TestDPOReferenceFree:
    def test_reference_free_loss(self):
        pw = torch.tensor([-2.0, -2.5])
        pl = torch.tensor([-3.0, -3.5])
        rw = torch.zeros(2)
        rl = torch.zeros(2)
        cfg = DPOConfig(reference_free=True)
        l, m = dpo_loss(pw, pl, rw, rl, cfg)
        assert l.item() >= 0


class TestCustomConstitution:
    def test_custom_constitution(self):
        cfg = ConstitutionalConfig(constitution=["Be brief.", "Be kind."])
        cai = ConstitutionalAI(cfg)
        cr  = cai.critique_and_revise("response")
        assert cr.principle in ["Be brief.", "Be kind."]


class TestKTOAllDesirable:
    def test_all_desirable(self):
        lp = torch.tensor([-2.0, -2.5, -2.3])
        rl = torch.tensor([-2.1, -2.4, -2.2])
        kl = torch.tensor(0.05)
        is_d = torch.tensor([True, True, True])
        l, m = kto_loss(lp, rl, is_d, kl, KTOConfig())
        assert l.item() >= 0

    def test_all_undesirable(self):
        lp = torch.tensor([-3.0, -3.5])
        rl = torch.tensor([-2.9, -3.3])
        kl = torch.tensor(0.05)
        is_d = torch.tensor([False, False])
        l, m = kto_loss(lp, rl, is_d, kl, KTOConfig())
        assert l.item() >= 0


class TestRMGradient:
    def test_gradients_flow(self):
        rm = RewardModel(TinyLM(), d_model=V)
        c  = torch.randint(0, V, (2, 4))
        r  = torch.randint(0, V, (2, 4))
        l, _ = rm.preference_loss(c, r)
        l.backward()
        has_grad = any(p.grad is not None for p in rm.parameters())
        assert has_grad


class TestKLComputation:
    def test_compute_kl_returns_float(self):
        rm = RewardModel(TinyLM(), V)
        ev = AlignmentEvaluator(rm, TinyLM())
        ids = torch.randint(0, V, (2, 6))
        kl  = ev.compute_kl(TinyLM(), ids)
        assert isinstance(kl, float)
