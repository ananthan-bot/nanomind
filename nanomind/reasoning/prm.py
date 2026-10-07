"""
nanomind/reasoning/prm.py — Process Reward Model (PRM800K style step-level verification).
"""
import math
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.reasoning.config import PRMConfig


class StepDelimiter:
    """Utility to split reasoning traces into atomic verification steps."""

    def __init__(self, delimiter: str = "\n\n"):
        self.delimiter = delimiter

    def split(self, text: str) -> List[str]:
        """Split reasoning text into steps, stripping empty segments."""
        raw = text.split(self.delimiter)
        steps = [s.strip() for s in raw if s.strip()]
        return steps if steps else [text.strip()]

    def join(self, steps: List[str]) -> str:
        """Recombine steps with the delimiter."""
        return self.delimiter.join(steps)


class ProcessRewardModel(nn.Module):
    """
    Process Reward Model that predicts correctness probabilities at each reasoning step.
    Evaluates tokens at step boundaries and projects hidden states to step probabilities.
    """

    def __init__(self, d_model: int = 128, num_classes: int = 2, config: Optional[PRMConfig] = None):
        super().__init__()
        self.config = config or PRMConfig(hidden_dim=d_model, num_classes=num_classes)
        self.d_model = d_model
        self.num_classes = num_classes

        # Step value / classification head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1 if num_classes == 2 else num_classes)
        )

    def forward(self, hidden_states: torch.Tensor, step_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        hidden_states: (B, T, D) or (T, D)
        step_indices: token indices corresponding to step delimiters.
        Returns:
            step_logits / step_probs: (num_steps,) or (B, num_steps)
        """
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)  # (1, T, D)

        B, T, D = hidden_states.shape

        if step_indices is None or len(step_indices) == 0:
            # Score the terminal token if no explicit step indices given
            step_indices = [T - 1]

        # Extract embeddings at step positions
        step_idx_tensor = torch.tensor(step_indices, device=hidden_states.device, dtype=torch.long)
        selected_states = hidden_states[:, step_idx_tensor, :]  # (B, num_steps, D)

        logits = self.value_head(selected_states).squeeze(-1)  # (B, num_steps)
        if self.num_classes == 2:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)

    def score_step_embeddings(self, step_embeddings: torch.Tensor) -> torch.Tensor:
        """Score pre-pooled step embeddings (num_steps, D). Returns probabilities in [0, 1]."""
        logits = self.value_head(step_embeddings).squeeze(-1)
        return torch.sigmoid(logits)


def aggregate_step_scores(scores: List[float], method: str = "product") -> float:
    """
    Aggregate step-level probabilities into an overall trajectory score.
    Methods:
        - 'product': P(all steps correct) = prod_t p_t
        - 'min': Weakest link principle = min_t p_t
        - 'last': Final outcome score = p_T
        - 'mean': Arithmetic average = mean_t p_t
    """
    if not scores:
        return 0.0

    if method == "product":
        prod = 1.0
        for s in scores:
            prod *= max(1e-7, float(s))
        return float(prod)
    elif method == "min":
        return float(min(scores))
    elif method == "last":
        return float(scores[-1])
    elif method == "mean":
        return float(sum(scores) / len(scores))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


class PRMLoss(nn.Module):
    """
    Binary Cross-Entropy Loss for PRM training at step boundaries.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, step_probs: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        step_probs: (B, num_steps) probabilities in (0, 1)
        targets: (B, num_steps) binary labels {0, 1}
        mask: (B, num_steps) boolean mask of valid steps
        """
        bce = F.binary_cross_entropy(step_probs.clamp(1e-7, 1 - 1e-7), targets.float(), reduction="none")
        if mask is not None:
            bce = bce * mask.float()
            if self.reduction == "mean":
                denom = mask.sum().clamp(min=1.0)
                return bce.sum() / denom
            elif self.reduction == "sum":
                return bce.sum()
        if self.reduction == "mean":
            return bce.mean()
        elif self.reduction == "sum":
            return bce.sum()
        return bce
