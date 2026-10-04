"""
nanomind/eval/leaderboard.py — Model leaderboard and comparison utilities.

Compare multiple models on the same benchmarks.
Generate ranking tables, Elo ratings, and radar charts.

## Elo Rating System

Used in LMSYS Chatbot Arena to rank LLMs:
  K = Elo update factor (default 32)
  Expected score: E_A = 1 / (1 + 10^((R_B - R_A)/400))
  Update: R_A += K × (S_A - E_A)  where S_A = 1 if A wins, 0.5 if tie, 0 if loss

Starting Elo: 1000 for all models
Higher Elo = better model.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ModelScore:
    """A model's score on a benchmark suite."""
    model_name:  str
    scores:      dict[str, float]   # task → accuracy
    elo:         float = 1000.0

    @property
    def mean_score(self) -> float:
        return sum(self.scores.values()) / max(len(self.scores), 1)

    def to_dict(self) -> dict:
        return {
            "model":      self.model_name,
            "mean_score": round(self.mean_score, 4),
            "elo":        round(self.elo, 1),
            **{k: round(v, 4) for k, v in self.scores.items()},
        }


class Leaderboard:
    """
    Model leaderboard with Elo-based ranking.

    Args:
        task_names: Names of evaluation tasks.

    Example::

        board = Leaderboard(["MMLU", "GSM8K", "HellaSwag"])
        board.add_model("GPT-4",     {"MMLU": 0.86, "GSM8K": 0.92, "HellaSwag": 0.96})
        board.add_model("LLaMA-70B", {"MMLU": 0.79, "GSM8K": 0.77, "HellaSwag": 0.87})
        print(board.ranking())
    """

    def __init__(self, task_names: list[str]) -> None:
        self.task_names = task_names
        self._models:  list[ModelScore] = []

    def add_model(self, name: str, scores: dict[str, float]) -> None:
        """Add a model's scores to the leaderboard."""
        self._models.append(ModelScore(model_name=name, scores=scores))

    def ranking(self) -> list[ModelScore]:
        """Return models sorted by mean score (descending)."""
        return sorted(self._models, key=lambda m: -m.mean_score)

    def update_elo(self, winner: str, loser: str, k: float = 32.0) -> None:
        """Update Elo ratings after a pairwise comparison."""
        def find(name):
            for m in self._models:
                if m.model_name == name:
                    return m
            return None

        w = find(winner)
        l = find(loser)
        if w is None or l is None:
            return

        ea = 1.0 / (1.0 + 10 ** ((l.elo - w.elo) / 400.0))
        eb = 1.0 - ea
        w.elo += k * (1.0 - ea)
        l.elo += k * (0.0 - eb)

    def leaderboard_table(self) -> list[dict]:
        """Return full leaderboard as list of dicts."""
        ranked = self.ranking()
        return [
            {"rank": i + 1, **m.to_dict()}
            for i, m in enumerate(ranked)
        ]

    def best_model(self) -> ModelScore | None:
        """Return the top-ranked model."""
        ranked = self.ranking()
        return ranked[0] if ranked else None

    def task_comparison(self, task: str) -> list[tuple[str, float]]:
        """Compare all models on a specific task."""
        result = [(m.model_name, m.scores.get(task, 0.0)) for m in self._models]
        return sorted(result, key=lambda x: -x[1])
