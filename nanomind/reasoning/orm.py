"""
nanomind/reasoning/orm.py — Outcome Reward Model (ORM), pairwise ranking loss, and calibration metrics.
"""
import math
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class OutcomeRewardModel(nn.Module):
    """
    Outcome Reward Model (ORM) that scores an entire solution at the final token.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, final_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        final_hidden_state: (B, D) terminal token representation.
        Returns: scalar reward (B,)
        """
        return self.score_head(final_hidden_state).squeeze(-1)


class MarginRankingLoss(nn.Module):
    """
    Pairwise ranking loss for trajectory preference optimization (Bradley-Terry formulation).
    Loss = -log sigmoid(reward_chosen - reward_rejected - margin)
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(self, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
        diff = chosen_rewards - rejected_rewards - self.margin
        return -F.logsigmoid(diff).mean()


def compute_brier_score(preds: List[float], targets: List[int]) -> float:
    """
    Compute Brier score: mean squared error of probabilistic predictions.
    Lower is better (0 = perfect calibration and discrimination).
    """
    if not preds or len(preds) != len(targets):
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(preds, targets)) / len(preds)


def compute_calibration_error(probs: List[float], labels: List[int], n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    Measures difference between predicted confidence and empirical accuracy across bins.
    """
    if not probs or len(probs) != len(labels):
        return 0.0

    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        b_idx = min(n_bins - 1, int(p * n_bins))
        bins[b_idx].append((p, y))

    ece = 0.0
    total = len(probs)
    for b in bins:
        if not b:
            continue
        bin_size = len(b)
        bin_conf = sum(item[0] for item in b) / bin_size
        bin_acc = sum(item[1] for item in b) / bin_size
        ece += (bin_size / total) * abs(bin_conf - bin_acc)

    return float(ece)


def compare_prm_vs_orm(step_scores: List[float], orm_score: float) -> Dict[str, float]:
    """
    Compare PRM step metrics against ORM terminal outcome score.
    Detects whether failure was sudden (weakest link) or uniform degradation.
    """
    from nanomind.reasoning.prm import aggregate_step_scores
    prm_prod = aggregate_step_scores(step_scores, "product")
    prm_min = aggregate_step_scores(step_scores, "min")
    prm_mean = aggregate_step_scores(step_scores, "mean")

    return {
        "prm_product": prm_prod,
        "prm_min": prm_min,
        "prm_mean": prm_mean,
        "orm_score": float(orm_score),
        "discrepancy": abs(prm_prod - float(orm_score))
    }
