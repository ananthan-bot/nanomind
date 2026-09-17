"""
nanomind/nas/search_space.py — NAS search space definitions.

## What is Neural Architecture Search?

NAS automates the design of neural network architectures.
Instead of hand-crafting d_model, n_heads, n_layers, dropout, etc.,
NAS searches over a defined space to find the optimal configuration.

## Search Space Types

Macro search: choose which layers to include (e.g., conv vs attn)
Micro search: choose per-layer hyperparameters (d_model, heads, FFN ratio)

NanoMind NAS searches the micro space of transformer hyperparameters:
  d_model:   [32, 64, 128, 256]
  n_layers:  [1, 2, 4, 6, 8]
  n_heads:   [1, 2, 4, 8]
  ffn_ratio: [1, 2, 4]
  dropout:   [0.0, 0.1, 0.2]

## Algorithms

Random Search (baseline):      sample uniformly from the space
Grid Search:                   exhaustive enumeration
Evolutionary Search:           mutation + selection over generations
Bayesian Optimisation:         model P(performance | config) with a GP
DARTS (Differentiable NAS):    relax discrete choices to continuous weights
Once-for-All / Weight Sharing: train a supernet, sample subnets

References:
  NASNet: Zoph & Le (2016) https://arxiv.org/abs/1611.01578
  DARTS: Liu et al. (2018) https://arxiv.org/abs/1806.09055
  Once-for-All: Cai et al. (2019) https://arxiv.org/abs/1908.09791
"""

from __future__ import annotations
import itertools
import random
from dataclasses import dataclass, field


@dataclass
class ArchConfig:
    """
    A single architecture configuration sampled from the search space.

    Attributes:
        d_model:   Token embedding dimension.
        n_layers:  Number of transformer blocks.
        n_heads:   Number of attention heads (must divide d_model).
        ffn_ratio: FFN hidden = ffn_ratio * d_model.
        dropout:   Dropout rate.
        max_seq:   Maximum sequence length.
    """
    d_model:   int   = 128
    n_layers:  int   = 4
    n_heads:   int   = 4
    ffn_ratio: int   = 4
    dropout:   float = 0.1
    max_seq:   int   = 256

    def __post_init__(self) -> None:
        assert self.d_model  % self.n_heads == 0,             f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.d_model   > 0
        assert self.n_layers  > 0
        assert self.n_heads   > 0
        assert self.ffn_ratio > 0
        assert 0.0 <= self.dropout < 1.0

    @property
    def n_params_estimate(self) -> int:
        """Rough parameter count estimate (embedding + transformer layers)."""
        embed     = self.d_model * 1000         # vocab embedding (vocab=1000)
        per_layer = (
            4 * self.d_model ** 2              # Q, K, V, O projections
            + 2 * self.d_model * self.d_model * self.ffn_ratio  # FFN
        )
        return embed + self.n_layers * per_layer

    def to_dict(self) -> dict:
        return {
            "d_model":   self.d_model,
            "n_layers":  self.n_layers,
            "n_heads":   self.n_heads,
            "ffn_ratio": self.ffn_ratio,
            "dropout":   self.dropout,
            "max_seq":   self.max_seq,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ArchConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SearchSpace:
    """
    Defines the discrete hyperparameter search space for NAS.

    Args:
        d_model_choices:   Candidate embedding dimensions.
        n_layers_choices:  Candidate layer counts.
        n_heads_choices:   Candidate head counts.
        ffn_ratio_choices: Candidate FFN expansion ratios.
        dropout_choices:   Candidate dropout rates.

    Example::

        space = SearchSpace()
        cfg   = space.random_sample()   # random ArchConfig
        all_  = space.grid()            # ALL valid combinations
        print(f"Space size: {space.size}")
    """

    def __init__(
        self,
        d_model_choices:   list[int]   = None,
        n_layers_choices:  list[int]   = None,
        n_heads_choices:   list[int]   = None,
        ffn_ratio_choices: list[int]   = None,
        dropout_choices:   list[float] = None,
    ) -> None:
        self.d_model_choices   = d_model_choices   or [32, 64, 128, 256]
        self.n_layers_choices  = n_layers_choices  or [1, 2, 4, 6]
        self.n_heads_choices   = n_heads_choices   or [1, 2, 4, 8]
        self.ffn_ratio_choices = ffn_ratio_choices or [1, 2, 4]
        self.dropout_choices   = dropout_choices   or [0.0, 0.1, 0.2]

    def _valid(self, d_model: int, n_heads: int) -> bool:
        return d_model % n_heads == 0

    def random_sample(self) -> ArchConfig:
        """Sample a random valid architecture config."""
        while True:
            d = random.choice(self.d_model_choices)
            h = random.choice(self.n_heads_choices)
            if self._valid(d, h):
                return ArchConfig(
                    d_model   = d,
                    n_layers  = random.choice(self.n_layers_choices),
                    n_heads   = h,
                    ffn_ratio = random.choice(self.ffn_ratio_choices),
                    dropout   = random.choice(self.dropout_choices),
                )

    def grid(self) -> list[ArchConfig]:
        """Generate all valid (d_model, n_heads) combinations."""
        configs = []
        for d, l, h, f, drop in itertools.product(
            self.d_model_choices, self.n_layers_choices,
            self.n_heads_choices, self.ffn_ratio_choices,
            self.dropout_choices,
        ):
            if self._valid(d, h):
                configs.append(ArchConfig(d_model=d, n_layers=l,
                                           n_heads=h, ffn_ratio=f, dropout=drop))
        return configs

    @property
    def size(self) -> int:
        """Number of valid architectures in the search space."""
        return len(self.grid())

    def neighbours(self, cfg: ArchConfig, n: int = 4) -> list[ArchConfig]:
        """
        Return N random neighbours of a config (one dimension changed).

        Used by local search and evolutionary mutation.
        """
        results = []
        dims = ["d_model", "n_layers", "n_heads", "ffn_ratio", "dropout"]
        seen = set()
        attempts = 0
        while len(results) < n and attempts < 100:
            attempts += 1
            dim = random.choice(dims)
            choices_map = {
                "d_model":   self.d_model_choices,
                "n_layers":  self.n_layers_choices,
                "n_heads":   self.n_heads_choices,
                "ffn_ratio": self.ffn_ratio_choices,
                "dropout":   self.dropout_choices,
            }
            new_val = random.choice(choices_map[dim])
            d       = cfg.to_dict()
            d[dim]  = new_val
            try:
                candidate = ArchConfig.from_dict(d)
                key = tuple(candidate.to_dict().values())
                if key not in seen:
                    seen.add(key)
                    results.append(candidate)
            except AssertionError:
                pass
        return results
