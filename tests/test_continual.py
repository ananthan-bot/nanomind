"""tests/test_continual.py — Tests for NanoMind continual learning."""
import copy
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.continual import (
    ContinualConfig, EWC, SynapticIntelligence,
    ReplayBuffer, ReplayEntry, PackNet,
    ContinualMetrics, ContinualEvaluator, ContinualTrainer,
)

V = 8

class TinyLM(nn.Module):
    def __init__(self, V=8, D=16, T=4):
        super().__init__()
        self.T   = T
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.lm  = nn.Linear(D, V, bias=False)
    def forward(self, x, t=None):
        B, S = x.shape
        h = self.tok(x) + self.pos(torch.arange(min(S, self.T)))
        h = h[:, :self.T]
        logits = self.lm(h)
        loss   = F.cross_entropy(logits.view(-1, V), t.view(-1)) if t is not None else None
        return logits, loss

def make_batches(n=4, T=4):
    return [(torch.randint(0, V, (2, T)), torch.randint(0, V, (2, T)))
            for _ in range(n)]


# ── ContinualConfig ───────────────────────────────────────────────────────────

class TestContinualConfig:
    def test_defaults(self):
        cfg = ContinualConfig()
        assert cfg.strategy == "ewc"
        assert cfg.n_tasks  == 5

    def test_invalid_strategy(self):
        with pytest.raises(AssertionError):
            ContinualConfig(strategy="unknown")

    def test_uses_regularisation(self):
        assert ContinualConfig(strategy="ewc").uses_regularisation
        assert ContinualConfig(strategy="si").uses_regularisation
        assert not ContinualConfig(strategy="replay").uses_regularisation

    def test_uses_replay(self):
        assert ContinualConfig(strategy="replay").uses_replay

    def test_uses_masking(self):
        assert ContinualConfig(strategy="packnet").uses_masking

    def test_invalid_prune_ratio(self):
        with pytest.raises(AssertionError):
            ContinualConfig(packnet_prune_ratio=1.0)


# ── EWC ───────────────────────────────────────────────────────────────────────

class TestEWC:
    def _ewc(self):
        m = TinyLM(V)
        return EWC(m, lambda_=10.0, n_samples=4), m

    def test_register_task(self):
        ewc, m = self._ewc()
        batches = make_batches()
        ewc.register_task(batches, task_id=0)
        assert ewc.n_tasks_registered == 1

    def test_penalty_zero_before_tasks(self):
        ewc, m = self._ewc()
        assert ewc.penalty().item() == 0.0

    def test_penalty_positive_after_register(self):
        ewc, m = self._ewc()
        ewc.register_task(make_batches(), 0)
        # Move weights slightly
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        pen = ewc.penalty()
        assert pen.item() >= 0.0

    def test_fisher_summary_keys(self):
        ewc, m = self._ewc()
        ewc.register_task(make_batches(), 0)
        s = ewc.fisher_summary(0)
        for k in ("task_id", "total_fisher", "n_params", "mean_fisher"):
            assert k in s

    def test_register_two_tasks(self):
        ewc, m = self._ewc()
        ewc.register_task(make_batches(), 0)
        ewc.register_task(make_batches(), 1)
        assert ewc.n_tasks_registered == 2


# ── SynapticIntelligence ──────────────────────────────────────────────────────

class TestSI:
    def _si(self):
        m  = TinyLM(V)
        si = SynapticIntelligence(m, lambda_=10.0)
        return si, m

    def test_begin_task_sets_state(self):
        si, m = self._si()
        si.begin_task()
        assert len(si._W) > 0

    def test_penalty_zero_before_tasks(self):
        si, m = self._si()
        assert si.penalty().item() == 0.0

    def test_end_task_increments(self):
        si, m = self._si()
        si.begin_task()
        x, y = make_batches(1)[0]
        _, loss = m(x, y)
        loss.backward()
        si.update_importances()
        si.end_task()
        assert si.n_tasks == 1

    def test_penalty_non_negative(self):
        si, m = self._si()
        si.begin_task()
        x, y = make_batches(1)[0]
        _, loss = m(x, y); loss.backward()
        si.update_importances()
        si.end_task()
        assert si.penalty().item() >= 0.0


# ── ReplayBuffer ──────────────────────────────────────────────────────────────

class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(max_size=10)
        x, y = torch.randint(0, V, (4,)), torch.randint(0, V, (4,))
        buf.add(x, y, task_id=0)
        assert len(buf) == 1

    def test_reservoir_caps_at_max(self):
        buf = ReplayBuffer(max_size=5)
        for _ in range(20):
            x = torch.randint(0, V, (4,))
            y = torch.randint(0, V, (4,))
            buf.add(x, y, task_id=0)
        assert len(buf) == 5

    def test_sample_returns_entries(self):
        buf = ReplayBuffer(max_size=10)
        for i in range(8):
            buf.add(torch.randint(0, V, (4,)), torch.randint(0, V, (4,)), task_id=i % 2)
        entries = buf.sample(3)
        assert len(entries) == 3

    def test_sample_batch_shape(self):
        buf = ReplayBuffer(max_size=10)
        for _ in range(6):
            buf.add(torch.randint(0, V, (4,)), torch.randint(0, V, (4,)), task_id=0)
        xs, ys = buf.sample_batch(4)
        assert xs.shape[0] <= 4
        assert ys.shape == xs.shape

    def test_task_counts(self):
        buf = ReplayBuffer(max_size=20)
        for i in range(10):
            buf.add(torch.randint(0, V, (4,)), torch.randint(0, V, (4,)), task_id=i % 2)
        counts = buf.task_counts()
        assert set(counts.keys()) <= {0, 1}

    def test_n_seen_increments(self):
        buf = ReplayBuffer(max_size=10)
        for _ in range(5):
            buf.add(torch.randint(0, V, (4,)), torch.randint(0, V, (4,)), task_id=0)
        assert buf.n_seen == 5


# ── PackNet ───────────────────────────────────────────────────────────────────

class TestPackNet:
    def _pn(self, ratio=0.3):
        m  = TinyLM(V)
        pn = PackNet(m, prune_ratio=ratio)
        return pn, m

    def test_prune_and_pack_returns_stats(self):
        pn, m = self._pn()
        stats = pn.prune_and_pack(task_id=0)
        assert "n_pruned" in stats and "n_total" in stats

    def test_n_tasks_increments(self):
        pn, m = self._pn()
        pn.prune_and_pack(0)
        pn.prune_and_pack(1)
        assert pn.n_tasks == 2

    def test_free_ratio_decreases(self):
        pn, m = self._pn(ratio=0.4)
        r0 = pn.free_ratio
        pn.prune_and_pack(0)
        r1 = pn.free_ratio
        assert r1 < r0

    def test_apply_mask_restores_weights(self):
        pn, m = self._pn()
        w_before = list(m.parameters())[0].data.clone()
        pn.prune_and_pack(0)
        with torch.no_grad():
            list(m.parameters())[0].add_(torch.ones_like(list(m.parameters())[0]))
        pn.apply_mask(0)
        w_after = list(m.parameters())[0].data
        assert torch.allclose(w_before, w_after)

    def test_invalid_task_raises(self):
        pn, m = self._pn()
        with pytest.raises(ValueError):
            pn.apply_mask(99)


# ── ContinualMetrics ──────────────────────────────────────────────────────────

class TestContinualMetrics:
    def _metrics(self):
        return ContinualMetrics(
            accuracy_matrix=[[0.9, 0.7, 0.6],
                              [0.0, 0.85, 0.75],
                              [0.0, 0.0, 0.8]],
            n_tasks=3,
        )

    def test_average_accuracy(self):
        m = self._metrics()
        # AA = (0.6 + 0.75 + 0.8) / 3
        assert abs(m.average_accuracy - (0.6 + 0.75 + 0.8) / 3) < 1e-4

    def test_backward_transfer_negative_forgetting(self):
        m = self._metrics()
        # BWT = ((0.6-0.9) + (0.75-0.85)) / 2 = (-0.3 + -0.1) / 2 = -0.2
        assert m.backward_transfer < 0.0   # forgetting

    def test_forgetting_positive(self):
        m = self._metrics()
        assert m.forgetting > 0.0

    def test_to_dict_keys(self):
        d = self._metrics().to_dict()
        for k in ("average_accuracy", "backward_transfer", "forgetting", "n_tasks"):
            assert k in d

    def test_single_task_zero_bwt(self):
        m = ContinualMetrics([[0.8]], n_tasks=1)
        assert m.backward_transfer == 0.0


# ── ContinualTrainer ──────────────────────────────────────────────────────────

class TestContinualTrainer:
    def _trainer(self, strategy="ewc"):
        m   = TinyLM(V)
        cfg = ContinualConfig(strategy=strategy, n_tasks=2, ewc_lambda=1.0,
                               ewc_n_samples=4, replay_buffer_size=20)
        return ContinualTrainer(m, cfg), m

    def test_train_task_returns_log(self):
        t, _ = self._trainer("naive")
        log  = t.train_task(0, make_batches(4), epochs=1)
        assert "task_id" in log and "final_loss" in log

    def test_ewc_trainer_registers(self):
        t, _ = self._trainer("ewc")
        t.train_task(0, make_batches(4), epochs=1)
        assert t._ewc.n_tasks_registered == 1

    def test_replay_trainer_fills_buffer(self):
        t, _ = self._trainer("replay")
        t.train_task(0, make_batches(4), epochs=1)
        assert len(t._replay) > 0

    def test_packnet_trainer_packs(self):
        t, _ = self._trainer("packnet")
        t.train_task(0, make_batches(4), epochs=1)
        assert t._packnet.n_tasks == 1

    def test_task_logs(self):
        t, _ = self._trainer("naive")
        t.train_task(0, make_batches(), epochs=1)
        t.train_task(1, make_batches(), epochs=1)
        assert len(t.task_logs) == 2


class TestEWCLambda:
    def test_higher_lambda_higher_penalty(self):
        m1 = TinyLM(V); ewc1 = EWC(m1, lambda_=10.0, n_samples=4)
        m2 = TinyLM(V); ewc2 = EWC(m2, lambda_=1000.0, n_samples=4)
        batches = make_batches()
        ewc1.register_task(batches, 0)
        ewc2.register_task(batches, 0)
        with torch.no_grad():
            for p in m1.parameters(): p.add_(torch.ones_like(p) * 0.1)
            for p in m2.parameters(): p.add_(torch.ones_like(p) * 0.1)
        # Copy m1's state to m2 for fair comparison
        m2.load_state_dict(m1.state_dict())
        assert ewc2.penalty().item() >= ewc1.penalty().item()


class TestReplayBatch:
    def test_add_batch(self):
        buf = ReplayBuffer(max_size=20)
        xs  = torch.randint(0, V, (4, 4))
        ys  = torch.randint(0, V, (4, 4))
        buf.add_batch(xs, ys, task_id=0)
        assert buf.n_seen == 4

    def test_add_with_logits(self):
        buf    = ReplayBuffer(max_size=10)
        x      = torch.randint(0, V, (4,))
        y      = torch.randint(0, V, (4,))
        logits = torch.randn(4, V)
        buf.add(x, y, task_id=0, logits=logits)
        assert buf._buffer[0].logits is not None


class TestPackNetFreeze:
    def test_freeze_zeros_frozen_grads(self):
        m  = TinyLM(V)
        pn = PackNet(m, prune_ratio=0.5)
        pn.prune_and_pack(0)   # mark some weights as frozen
        # Compute grads
        x, y = make_batches(1)[0]
        _, loss = m(x, y); loss.backward()
        pn.freeze_past_weights()
        # Frozen weights should have zero gradient
        for name, param in m.named_parameters():
            if param.grad is not None and name in pn._frozen_mask:
                frozen_grads = param.grad[pn._frozen_mask[name]]
                assert (frozen_grads == 0.0).all()


class TestNoForgetting:
    def test_no_forgetting_perfect(self):
        """If accuracy stays constant, forgetting should be 0."""
        m = ContinualMetrics(
            accuracy_matrix=[[0.9, 0.9, 0.9],
                              [0.0, 0.85, 0.85],
                              [0.0, 0.0, 0.8]],
            n_tasks=3,
        )
        assert m.forgetting == 0.0


class TestSIPenaltyGrad:
    def test_si_penalty_gradient(self):
        m  = TinyLM(V)
        si = SynapticIntelligence(m, lambda_=10.0)
        si.begin_task()
        x, y = make_batches(1)[0]
        _, loss = m(x, y); loss.backward()
        si.update_importances()
        si.end_task()
        pen = si.penalty()
        pen.backward()
        has_grad = any(p.grad is not None for p in m.parameters())
        assert has_grad


class TestReplaySmallBuffer:
    def test_sample_fewer_than_requested(self):
        buf = ReplayBuffer(max_size=3)
        for _ in range(3):
            buf.add(torch.randint(0, V, (4,)), torch.randint(0, V, (4,)), 0)
        entries = buf.sample(10)
        assert len(entries) == 3   # capped at buffer size


class TestSITrainer:
    def test_si_train_two_tasks(self):
        m   = TinyLM(V)
        cfg = ContinualConfig(strategy="si", n_tasks=2, ewc_lambda=1.0)
        t   = ContinualTrainer(m, cfg)
        log0 = t.train_task(0, make_batches(4), epochs=1)
        log1 = t.train_task(1, make_batches(4), epochs=1)
        assert log0["strategy"] == "si"
        assert log1["strategy"] == "si"
        assert t._si.n_tasks == 2
