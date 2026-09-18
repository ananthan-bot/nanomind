"""
nanomind/interpret/attribution.py — Token contribution attribution.

Decomposes the model output into contributions from each input token.

Methods:
  1. LIME-style local approximation:
       Mask random subsets of tokens → fit linear model on output changes
  2. Shapley values (approximate):
       Average marginal contribution over random token orderings
  3. Occlusion / leave-one-out:
       Mask each token and measure output change

Shapley values have desirable axioms:
  - Efficiency: contributions sum to total output
  - Symmetry: equal tokens get equal attribution
  - Dummy: irrelevant tokens get zero
  - Linearity: additive for combined models

Reference:
  Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions (SHAP)"
  https://arxiv.org/abs/1705.07874
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import random
from dataclasses import dataclass


@dataclass
class Attribution:
    """Token attribution scores."""
    scores:  list[float]    # per-token contribution
    tokens:  list[str]
    method:  str
    baseline: float

    def normalised(self) -> list[float]:
        """Normalise to sum to 1."""
        total = sum(abs(s) for s in self.scores) or 1.0
        return [s / total for s in self.scores]

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        """Return top-K contributing tokens."""
        pairs = sorted(zip(self.tokens, self.scores),
                       key=lambda x: abs(x[1]), reverse=True)
        return [(t, round(s, 4)) for t, s in pairs[:k]]

    def to_dict(self) -> dict:
        return {
            "method":   self.method,
            "tokens":   self.tokens,
            "scores":   [round(s, 4) for s in self.scores],
            "top_3":    self.top_k(3),
        }


class OcclusionAttributor:
    """
    Leave-one-out (occlusion) token attribution.

    Mask each token with a pad_id and measure change in target logit.

    Args:
        model:  Language model.
        pad_id: Token ID to use as mask (default: 0).

    Example::

        attr   = OcclusionAttributor(model)
        result = attr.attribute(input_ids, target_pos=5, target_class=42)
    """

    def __init__(self, model, pad_id: int = 0) -> None:
        self.model  = model
        self.pad_id = pad_id

    @torch.no_grad()
    def _score(self, ids: torch.Tensor, pos: int, cls: int) -> float:
        logits, _ = self.model(ids)
        return F.softmax(logits[0, pos], dim=-1)[cls].item()

    @torch.no_grad()
    def attribute(
        self,
        input_ids:    torch.Tensor,
        target_pos:   int,
        target_class: int,
        tokens:       list[str] = None,
    ) -> Attribution:
        """
        Compute occlusion attribution.

        Returns:
            :class:`Attribution` with per-token scores.
        """
        base_score = self._score(input_ids, target_pos, target_class)
        T          = input_ids.shape[1]
        scores     = []
        for t in range(T):
            masked       = input_ids.clone()
            masked[0, t] = self.pad_id
            score_t      = self._score(masked, target_pos, target_class)
            scores.append(base_score - score_t)  # positive = helpful token
        return Attribution(
            scores   = scores,
            tokens   = tokens or [str(i) for i in range(T)],
            method   = "occlusion",
            baseline = base_score,
        )


class ShapleyAttributor:
    """
    Approximate Shapley value attribution via random permutation sampling.

    Args:
        model:     Language model.
        pad_id:    Mask token ID.
        n_samples: Number of random orderings to average over.
    """

    def __init__(self, model, pad_id: int = 0, n_samples: int = 10) -> None:
        self.model     = model
        self.pad_id    = pad_id
        self.n_samples = n_samples

    @torch.no_grad()
    def _score(self, ids, pos, cls):
        logits, _ = self.model(ids)
        return F.softmax(logits[0, pos], dim=-1)[cls].item()

    @torch.no_grad()
    def attribute(
        self,
        input_ids:    torch.Tensor,
        target_pos:   int,
        target_class: int,
        tokens:       list[str] = None,
    ) -> Attribution:
        """Approximate Shapley values."""
        T      = input_ids.shape[1]
        phi    = [0.0] * T

        for _ in range(self.n_samples):
            order = list(range(T))
            random.shuffle(order)
            masked = torch.full_like(input_ids, self.pad_id)
            prev_score = self._score(masked, target_pos, target_class)

            for t in order:
                masked       = masked.clone()
                masked[0, t] = input_ids[0, t]
                curr_score   = self._score(masked, target_pos, target_class)
                phi[t]       += (curr_score - prev_score)
                prev_score   = curr_score

        phi = [v / self.n_samples for v in phi]
        return Attribution(
            scores   = phi,
            tokens   = tokens or [str(i) for i in range(T)],
            method   = "shapley",
            baseline = self._score(torch.full_like(input_ids, self.pad_id),
                                    target_pos, target_class),
        )
