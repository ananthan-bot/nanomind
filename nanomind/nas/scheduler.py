"""
nanomind/nas/scheduler.py — Progressive shrinking & hyperparameter scheduling.

Progressive Shrinking (Once-for-All, Cai et al.):
  Train a large supernet once, then "shrink" it by sampling
  smaller subnetworks. Fine-tuning each subnet takes seconds
  instead of hours.

  Shrinking order: elastic resolution → elastic depth → elastic width

NanoMind implements a simplified HP scheduler that progressively
narrows the search space from wide to narrow over search rounds.

Also implements:
  - Cosine annealing for evolutionary temperature
  - Warm restarts for exploration
"""

from __future__ import annotations
import math
from nanomind.nas.search_space import SearchSpace


class ProgressiveScheduler:
    """
    Progressively narrows the search space over rounds.

    Starts with the full space, then eliminates poor regions.
    Used to focus later search stages on promising sub-spaces.

    Args:
        space:          Initial search space.
        n_rounds:       Total search rounds.
        shrink_ratio:   Fraction of choices to keep per round.

    Example::

        sched = ProgressiveScheduler(space, n_rounds=4, shrink_ratio=0.5)
        for round_ in range(4):
            current_space = sched.get_space(round_)
            # search in current_space ...
    """

    def __init__(
        self,
        space:       SearchSpace,
        n_rounds:    int   = 4,
        shrink_ratio: float = 0.7,
    ) -> None:
        self.base_space   = space
        self.n_rounds     = n_rounds
        self.shrink_ratio = shrink_ratio

    def get_space(self, round_idx: int) -> SearchSpace:
        """
        Return the search space for a given round (progressively smaller).

        Args:
            round_idx: 0-indexed round number.

        Returns:
            :class:`SearchSpace` with reduced choices.
        """
        # Progressively keep fewer options in larger dimensions
        keep = max(1, int(len(self.base_space.d_model_choices)
                          * (self.shrink_ratio ** round_idx)))
        d_choices = sorted(self.base_space.d_model_choices, reverse=True)[:keep]

        keep_l = max(1, int(len(self.base_space.n_layers_choices)
                            * (self.shrink_ratio ** round_idx)))
        l_choices = sorted(self.base_space.n_layers_choices, reverse=True)[:keep_l]

        return SearchSpace(
            d_model_choices   = d_choices,
            n_layers_choices  = l_choices,
            n_heads_choices   = self.base_space.n_heads_choices,
            ffn_ratio_choices = self.base_space.ffn_ratio_choices,
            dropout_choices   = self.base_space.dropout_choices,
        )

    def temperature(self, round_idx: int) -> float:
        """
        Cosine annealing temperature for exploration (1.0 → 0.0).

        High temperature = more exploration (random).
        Low temperature  = more exploitation (greedy).
        """
        return 0.5 * (1 + math.cos(math.pi * round_idx / max(self.n_rounds - 1, 1)))


class WarmRestartScheduler:
    """
    Warm restart schedule for evolutionary search temperature.

    Args:
        T_0:    Initial period.
        T_mult: Period multiplier after each restart.
    """

    def __init__(self, T_0: int = 5, T_mult: int = 2) -> None:
        self.T_0    = T_0
        self.T_mult = T_mult
        self._step  = 0
        self._T_cur = T_0

    def step(self) -> float:
        """Advance one step and return current temperature."""
        t = self._step % self._T_cur
        temp = 0.5 * (1 + math.cos(math.pi * t / self._T_cur))
        self._step += 1
        if self._step >= self._T_cur:
            self._step  = 0
            self._T_cur *= self.T_mult
        return temp

    def reset(self) -> None:
        self._step  = 0
        self._T_cur = self.T_0
