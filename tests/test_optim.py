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
