"""
nanomind/alignment/preference.py — Preference data structures.

## From RLHF to Preference Learning

Classic RLHF (Day 11) uses a reward model + PPO loop.
Modern alignment methods use preference data directly:
  - DPO:  learns directly from (chosen, rejected) pairs — no RM!
  - IPO:  identity preference optimisation (fixes DPO over-fitting)
  - KTO:  learns from (prompt, response, good/bad) labels
  - RLAIF: AI feedback instead of human feedback (Constitutional AI)

## Preference Triplet

Standard preference dataset:
  (prompt, chosen, rejected)
  chosen:   the better response (human-preferred)
  rejected: the worse response (human-rejected)

Used in: Anthropic HH-RLHF, OpenAssistant, UltraFeedback.

## Constitutional AI (Bai et al., 2022)

Instead of human labellers, use the model itself to generate preferences:
  1. Model generates response to a harmful prompt
  2. Model critiques its own response using a CONSTITUTION
     (a list of principles: "be helpful, harmless, honest")
  3. Model revises its response based on the critique
  4. Use (original, revised) as preference pair → train with RL

This allows scalable alignment without human labellers.
Used by: Claude (Anthropic).

Reference:
  Bai et al. (2022) "Constitutional AI: Harmlessness from AI Feedback"
  https://arxiv.org/abs/2212.06950

  Rafailov et al. (2023) DPO:
  https://arxiv.org/abs/2305.18290
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PreferencePair:
    """A single preference pair: (prompt, chosen, rejected)."""
    prompt:   str
    chosen:   str
    rejected: str
    score_chosen:   float | None = None   # optional reward score
    score_rejected: float | None = None
    source:   str = "human"               # human | ai | synthetic

    @property
    def margin(self) -> float | None:
        if self.score_chosen and self.score_rejected:
            return self.score_chosen - self.score_rejected
        return None

    def to_dict(self) -> dict:
        return {
            "prompt":   self.prompt,
            "chosen":   self.chosen,
            "rejected": self.rejected,
            "source":   self.source,
        }

    def flip(self) -> "PreferencePair":
        """Flip chosen/rejected (for data augmentation / sanity check)."""
        return PreferencePair(
            prompt        = self.prompt,
            chosen        = self.rejected,
            rejected      = self.chosen,
            score_chosen  = self.score_rejected,
            score_rejected= self.score_chosen,
            source        = self.source,
        )


@dataclass
class BinaryFeedback:
    """Single response with binary good/bad label (for KTO)."""
    prompt:   str
    response: str
    is_good:  bool        # True = desirable, False = undesirable
    source:   str = "human"


class PreferenceDataset:
    """
    Dataset of preference pairs for alignment training.

    Args:
        pairs: Initial list of :class:`PreferencePair`.

    Example::

        dataset = PreferenceDataset()
        dataset.add(PreferencePair("What is 2+2?", "4", "fish"))
        print(f"Size: {len(dataset)}")
        for pair in dataset:
            ...
    """

    def __init__(self, pairs: list[PreferencePair] | None = None) -> None:
        self._pairs: list[PreferencePair] = list(pairs or [])

    def add(self, pair: PreferencePair) -> None:
        self._pairs.append(pair)

    def filter_by_margin(self, min_margin: float) -> "PreferenceDataset":
        """Keep only pairs where chosen is clearly better."""
        filtered = [p for p in self._pairs
                    if p.margin is not None and p.margin >= min_margin]
        return PreferenceDataset(filtered)

    def filter_by_source(self, source: str) -> "PreferenceDataset":
        return PreferenceDataset([p for p in self._pairs if p.source == source])

    def stats(self) -> dict:
        margins = [p.margin for p in self._pairs if p.margin is not None]
        return {
            "n_pairs":     len(self._pairs),
            "n_human":     sum(1 for p in self._pairs if p.source == "human"),
            "n_ai":        sum(1 for p in self._pairs if p.source == "ai"),
            "mean_margin": round(sum(margins) / len(margins), 4) if margins else None,
        }

    def to_binary(self) -> list[BinaryFeedback]:
        """Convert to binary feedback list (for KTO)."""
        out = []
        for p in self._pairs:
            out.append(BinaryFeedback(p.prompt, p.chosen,   is_good=True))
            out.append(BinaryFeedback(p.prompt, p.rejected, is_good=False))
        return out

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self):
        return iter(self._pairs)

    def __getitem__(self, idx):
        return self._pairs[idx]
