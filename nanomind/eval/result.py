"""
nanomind/eval/result.py — Structured evaluation result container.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from nanomind.eval.metrics import perplexity, bits_per_character


@dataclass
class EvalResult:
    """
    Container for all evaluation metrics.

    Attributes:
        loss:         Mean cross-entropy loss.
        perplexity:   exp(loss) — lower is better.
        bpc:          Bits-per-character.
        accuracy:     Top-1 token prediction accuracy.
        top_k_acc:    Top-K token prediction accuracy.
        n_batches:    Number of batches evaluated.
        n_tokens:     Total tokens evaluated.
    """

    loss:      float = float("nan")
    ppl:       float = float("nan")
    bpc:       float = float("nan")
    accuracy:  float = float("nan")
    top_k_acc: float = float("nan")
    n_batches: int   = 0
    n_tokens:  int   = 0

    @classmethod
    def from_loss(
        cls,
        loss: float,
        accuracy: float = float("nan"),
        top_k_acc: float = float("nan"),
        n_batches: int = 0,
        n_tokens: int = 0,
    ) -> "EvalResult":
        """Build an EvalResult from a loss value."""
        return cls(
            loss=loss,
            ppl=perplexity(loss),
            bpc=bits_per_character(loss),
            accuracy=accuracy,
            top_k_acc=top_k_acc,
            n_batches=n_batches,
            n_tokens=n_tokens,
        )

    def __str__(self) -> str:
        lines = [
            f"loss={self.loss:.4f}",
            f"ppl={self.ppl:.2f}",
            f"bpc={self.bpc:.4f}",
        ]
        if not (self.accuracy != self.accuracy):   # not nan
            lines.append(f"acc={self.accuracy:.4f}")
        if not (self.top_k_acc != self.top_k_acc):
            lines.append(f"top_k_acc={self.top_k_acc:.4f}")
        return "EvalResult(" + " | ".join(lines) + ")"
