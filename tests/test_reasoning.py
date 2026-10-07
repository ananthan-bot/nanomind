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
