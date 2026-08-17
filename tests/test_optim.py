"""
tests/test_optim.py — Tests for NanoMind optimizers and LR schedules.
"""

import pytest
import math
import torch

from nanomind.model import NanoMind, ModelConfig
from nanomind.optim import (
    get_optimizer,
    get_param_groups,
    count_param_groups,
    get_lr_scheduler,
    list_schedules,
    WarmupCosine,
    CosineDecay,
    LinearWarmup,
    LinearDecay,
    WarmupLinear,
    ConstantLR,
    compute_grad_norm,
    get_grad_stats,
    optimizer_summary,
    schedule_preview,
)

CFG = ModelConfig(
    vocab_size=32, block_size=8,
    d_model=32, n_layers=2, n_heads=2, dropout=0.0
)


@pytest.fixture
def model() -> NanoMind:
    torch.manual_seed(0)
    return NanoMind(CFG)


# ── Param groups ──────────────────────────────────────────────────────────────

class TestParamGroups:
    def test_returns_two_groups(self, model):
        groups = get_param_groups(model)
        assert len(groups) == 2

    def test_decay_group_has_weight_decay(self, model):
        groups = get_param_groups(model, weight_decay=0.1)
        assert groups[0]["weight_decay"] == 0.1

    def test_no_decay_group_has_zero_wd(self, model):
        groups = get_param_groups(model)
        assert groups[1]["weight_decay"] == 0.0

    def test_all_params_covered(self, model):
        groups  = get_param_groups(model)
        counts  = count_param_groups(groups)
        total   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        grouped = counts["decay"] + counts["no_decay"]
        assert grouped == total

    def test_no_duplicates(self, model):
        groups = get_param_groups(model)
        all_ids = [id(p) for g in groups for p in g["params"]]
        assert len(all_ids) == len(set(all_ids))


# ── CosineDecay ───────────────────────────────────────────────────────────────

class TestCosineDecay:
    def test_starts_at_max_lr(self):
        s = CosineDecay(max_lr=1e-3, min_lr=1e-5, total_steps=100)
        assert abs(s(0) - 1e-3) < 1e-9

    def test_ends_at_min_lr(self):
        s = CosineDecay(max_lr=1e-3, min_lr=1e-5, total_steps=100)
        assert abs(s(100) - 1e-5) < 1e-9

    def test_monotonically_decreasing(self):
        s = CosineDecay(max_lr=1e-3, min_lr=1e-5, total_steps=100)
        lrs = [s(i) for i in range(101)]
        assert all(lrs[i] >= lrs[i+1] for i in range(100))

    def test_midpoint_near_average(self):
        s = CosineDecay(max_lr=1e-3, min_lr=1e-5, total_steps=100)
        mid = s(50)
        avg = (1e-3 + 1e-5) / 2
        assert abs(mid - avg) < 1e-5


# ── LinearWarmup ──────────────────────────────────────────────────────────────

class TestLinearWarmup:
    def test_starts_near_zero(self):
        s = LinearWarmup(max_lr=1e-3, warmup_steps=10)
        assert s(0) < 1e-3

    def test_reaches_max_at_warmup_end(self):
        s = LinearWarmup(max_lr=1e-3, warmup_steps=10)
        assert abs(s(9) - 1e-3) < 1e-9

    def test_monotonically_increasing_during_warmup(self):
        s = LinearWarmup(max_lr=1e-3, warmup_steps=10)
        lrs = [s(i) for i in range(10)]
        assert all(lrs[i] < lrs[i+1] for i in range(9))

    def test_constant_after_warmup_no_schedule(self):
        s = LinearWarmup(max_lr=1e-3, warmup_steps=10)
        assert s(10) == 1e-3
        assert s(100) == 1e-3


# ── WarmupCosine ──────────────────────────────────────────────────────────────

class TestWarmupCosine:
    def test_starts_near_zero(self):
        s = WarmupCosine(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
        assert s(0) < 1e-3

    def test_reaches_max_at_warmup_end(self):
        s = WarmupCosine(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
        assert abs(s(9) - 1e-3) < 1e-9

    def test_ends_at_min_lr(self):
        s = WarmupCosine(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
        assert abs(s(100) - 1e-5) < 1e-9

    def test_overall_shape(self):
        s    = WarmupCosine(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
        lrs  = [s(i) for i in range(101)]
        # Warmup phase: increasing
        assert lrs[0] < lrs[9]
        # Post-warmup: decreasing
        assert lrs[10] > lrs[100]


# ── Factory ───────────────────────────────────────────────────────────────────

class TestScheduleFactory:
    def test_get_constant(self):
        s = get_lr_scheduler("constant", lr=1e-3)
        assert isinstance(s, ConstantLR)
        assert s(0) == 1e-3
        assert s(999) == 1e-3

    def test_get_cosine(self):
        s = get_lr_scheduler("cosine", max_lr=1e-3, min_lr=1e-5, total_steps=100)
        assert isinstance(s, CosineDecay)

    def test_get_warmup_cosine(self):
        s = get_lr_scheduler("warmup_cosine", max_lr=1e-3, min_lr=1e-5,
                             warmup_steps=10, total_steps=100)
        assert isinstance(s, WarmupCosine)

    def test_get_linear(self):
        s = get_lr_scheduler("linear", max_lr=1e-3, min_lr=1e-5, total_steps=100)
        assert isinstance(s, LinearDecay)

    def test_get_warmup_linear(self):
        s = get_lr_scheduler("warmup_linear", max_lr=1e-3, min_lr=1e-5,
                             warmup_steps=10, total_steps=100)
        assert isinstance(s, WarmupLinear)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_lr_scheduler("nosuchthing", lr=1e-3)

    def test_list_schedules(self):
        names = list_schedules()
        assert "warmup_cosine" in names
        assert "cosine" in names


# ── Grad utilities ────────────────────────────────────────────────────────────

class TestGradUtils:
    def test_grad_norm_zero_before_backward(self, model):
        assert compute_grad_norm(model) == 0.0

    def test_grad_norm_positive_after_backward(self, model):
        idx  = torch.randint(0, CFG.vocab_size, (2, 8))
        tgt  = torch.randint(0, CFG.vocab_size, (2, 8))
        _, loss = model(idx, tgt)
        loss.backward()
        assert compute_grad_norm(model) > 0.0

    def test_grad_stats_keys(self, model):
        idx  = torch.randint(0, CFG.vocab_size, (2, 8))
        tgt  = torch.randint(0, CFG.vocab_size, (2, 8))
        _, loss = model(idx, tgt)
        loss.backward()
        stats = get_grad_stats(model)
        assert "max" in stats and "l2_norm" in stats


# ── get_optimizer ─────────────────────────────────────────────────────────────

class TestGetOptimizer:
    def test_returns_adamw(self, model):
        opt = get_optimizer(model, optimizer_type="adamw")
        assert isinstance(opt, torch.optim.AdamW)

    def test_returns_sgd(self, model):
        opt = get_optimizer(model, optimizer_type="sgd")
        assert isinstance(opt, torch.optim.SGD)

    def test_unknown_raises(self, model):
        with pytest.raises(ValueError):
            get_optimizer(model, optimizer_type="rmsprop")

    def test_two_param_groups(self, model):
        opt = get_optimizer(model)
        assert len(opt.param_groups) == 2


# ── Summary utilities ─────────────────────────────────────────────────────────

class TestSummaryUtils:
    def test_optimizer_summary_string(self, model):
        opt = get_optimizer(model)
        s   = optimizer_summary(opt)
        assert "AdamW" in s
        assert "Group" in s

    def test_schedule_preview_length(self):
        s    = WarmupCosine(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
        pts  = schedule_preview(s, total_steps=100, n_points=5)
        assert len(pts) == 5
        assert pts[0][0] == 0
        assert pts[-1][0] == 100
