"""
nanomind/continual/config.py — Continual learning configuration.

## Catastrophic Forgetting

When a neural network learns task B after task A, it overwrites weights
that were critical for A — this is catastrophic forgetting (McCloskey & Cohen, 1989).

## Strategies to Prevent Forgetting

1. Regularisation (Bayesian):
   EWC (Kirkpatrick et al., 2017):   penalise changes to important weights
   SI  (Zenke et al., 2017):         online importance via path integrals
   LwF (Li & Hoiem, 2017):           distill old task outputs on new data

2. Replay / Memory:
   Experience Replay:   store subset of old data, replay during training
   Generative Replay:   train a generator to produce old-task samples
   DER++ (Buzzega et al., 2020): replay with dark experience distillation

3. Parameter Isolation:
   PackNet (Mallya & Lazebnik, 2018): prune and pack task masks
   Progressive Nets (Rusu et al., 2016): add new columns per task
   HAT (Serra et al., 2018): hard attention task masks

4. Architectural:
   PNN (Progressive Neural Networks): lateral connections
   DynamicExpander: grow new layers per task

References:
  Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"
  https://arxiv.org/abs/1612.00796

  van de Ven & Tolias (2019) survey: https://arxiv.org/abs/1904.07734
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ContinualConfig:
    """
    Configuration for a continual learning experiment.

    Attributes:
        strategy:       Strategy: ``"ewc"``, ``"replay"``, ``"packnet"``, ``"naive"``.
        n_tasks:        Total number of sequential tasks.
        ewc_lambda:     EWC regularisation strength.
        ewc_n_samples:  Samples for Fisher Information Matrix computation.
        replay_buffer_size: Max samples per task in replay buffer.
        replay_fraction:    Fraction of each batch from replay.
        packnet_prune_ratio: Fraction of weights to prune per task.
        si_xi:          SI damping factor.
        task_agnostic:  If True, task ID not given at test time.
    """
    strategy:            str   = "ewc"
    n_tasks:             int   = 5
    ewc_lambda:          float = 1000.0
    ewc_n_samples:       int   = 200
    replay_buffer_size:  int   = 500
    replay_fraction:     float = 0.3
    packnet_prune_ratio: float = 0.5
    si_xi:               float = 1e-3
    task_agnostic:       bool  = True

    def __post_init__(self) -> None:
        assert self.strategy in ("ewc", "replay", "packnet", "naive", "si", "lwf")
        assert self.n_tasks             >= 1
        assert self.ewc_lambda          >= 0.0
        assert self.ewc_n_samples       >= 1
        assert self.replay_buffer_size  >= 0
        assert 0.0 <= self.replay_fraction <= 1.0
        assert 0.0 <  self.packnet_prune_ratio < 1.0

    @property
    def uses_regularisation(self) -> bool:
        return self.strategy in ("ewc", "si", "lwf")

    @property
    def uses_replay(self) -> bool:
        return self.strategy == "replay"

    @property
    def uses_masking(self) -> bool:
        return self.strategy == "packnet"
