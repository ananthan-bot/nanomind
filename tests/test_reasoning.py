"""
tests/test_reasoning.py — Comprehensive unit tests for Reasoning & Test-Time Compute.
"""
import math
import pytest
import torch
from nanomind.reasoning import (
    PRMConfig,
    StepDelimiter,
    ProcessRewardModel,
    aggregate_step_scores,
    PRMLoss,
    OutcomeRewardModel,
    MarginRankingLoss,
    compute_brier_score,
    compute_calibration_error,
    compare_prm_vs_orm,
    ReasoningNode,
    puct_score,
    MonteCarloTreeSearch,
    TreeOfThoughts,
    StepBeamSearch,
    BestOfNVerifier,
    SelfReflectiveReasoner,
    compute_group_advantages,
    grpo_loss,
    GRPOTrainer,
    STaR,
    pass_at_k,
    estimate_test_time_flops,
    TestTimeScalingSimulator,
)


class TestPRM:
    def test_step_delimiter(self):
        delim = StepDelimiter(delimiter="\n\n")
        text = "Step 1: start\n\nStep 2: middle\n\nStep 3: end"
        steps = delim.split(text)
        assert len(steps) == 3
        assert steps[0] == "Step 1: start"
        rejoined = delim.join(steps)
        assert rejoined == text

    def test_prm_forward(self):
        prm = ProcessRewardModel(d_model=32, num_classes=2)
        h = torch.randn(2, 10, 32)
        step_indices = [3, 7, 9]
        probs = prm(h, step_indices=step_indices)
        assert probs.shape == (2, 3)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_aggregate_scores(self):
        scores = [0.9, 0.8, 0.5]
        prod = aggregate_step_scores(scores, "product")
        assert math.isclose(prod, 0.9 * 0.8 * 0.5, rel_tol=1e-5)

        min_val = aggregate_step_scores(scores, "min")
        assert min_val == 0.5

        mean_val = aggregate_step_scores(scores, "mean")
        assert math.isclose(mean_val, (0.9 + 0.8 + 0.5) / 3, rel_tol=1e-5)

    def test_prm_loss(self):
        loss_fn = PRMLoss()
        probs = torch.tensor([[0.8, 0.2]])
        targets = torch.tensor([[1, 0]])
        loss = loss_fn(probs, targets)
        assert loss.item() > 0.0


class TestORM:
    def test_orm_forward(self):
        orm = OutcomeRewardModel(d_model=32)
        h = torch.randn(4, 32)
        scores = orm(h)
        assert scores.shape == (4,)

    def test_margin_ranking_loss(self):
        loss_fn = MarginRankingLoss(margin=0.5)
        chosen = torch.tensor([2.0, 1.5])
        rejected = torch.tensor([0.5, 0.0])
        loss = loss_fn(chosen, rejected)
        assert loss.item() > 0.0

    def test_brier_and_calibration(self):
        preds = [0.9, 0.8, 0.1, 0.2]
        targets = [1, 1, 0, 0]
        brier = compute_brier_score(preds, targets)
        assert brier < 0.05  # Highly calibrated and accurate

        ece = compute_calibration_error(preds, targets, n_bins=5)
        assert 0.0 <= ece <= 1.0
