"""
nanomind/distributed/pipeline.py — Pipeline Parallelism (GPipe / 1F1B schedule).

## Pipeline Parallelism

Split model layers across GPUs (pipeline stages):
  GPU 0: Embedding + layers 0-7    (Stage 0)
  GPU 1: Layers 8-15               (Stage 1)
  GPU 2: Layers 16-23              (Stage 2)
  GPU 3: Layers 24-31 + LM head   (Stage 3)

Data flows through the pipeline as micro-batches.

## GPipe Schedule

Global batch B split into M micro-batches of size B/M:
  Stage 0: forward(μ0), forward(μ1), ..., forward(μM)
  Stage 1: waits, forward(μ0), forward(μ1), ...
  ...
  Backward in reverse order

Problem: "pipeline bubble" = idle time waiting for pipeline to fill/drain
  Bubble fraction = (p-1) / (M + p-1)  where p = pipeline stages

## 1F1B Schedule (PipeDream / Megatron-LM)

1 Forward, 1 Backward — interleaved:
  - Start backward as soon as first micro-batch completes forward
  - Reduces memory: only need to store activations for 1 micro-batch at a time
  - Bubble fraction = (p-1) / (M + p-1) (same as GPipe but memory efficient)

## Interleaved 1F1B (Virtual Stages)

Further reduce bubble by assigning multiple non-contiguous chunks per GPU:
  GPU 0: Layers 0-3, 16-19   (2 virtual stages)
  GPU 1: Layers 4-7, 20-23
  ...

References:
  Huang et al. (2019) GPipe: https://arxiv.org/abs/1811.06965
  Narayanan et al. (2021) 1F1B: https://arxiv.org/abs/2104.04473
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from nanomind.distributed.world import WorldConfig


@dataclass
class PipelineStage:
    """A single pipeline stage owning a set of layers."""
    stage_id:   int
    layers:     nn.ModuleList
    is_first:   bool   = False
    is_last:    bool   = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class MicroBatch:
    """A single micro-batch in the pipeline."""
    micro_id:    int
    activations: torch.Tensor | None = None
    gradients:   torch.Tensor | None = None
    is_forward:  bool = True


@dataclass
class PipelineSchedule:
    """
    Pipeline schedule descriptor.

    Tracks the sequence of (forward/backward, micro-batch, stage) steps
    for each pipeline rank.
    """
    n_microbatches: int
    n_stages:       int
    schedule_type:  str   = "1f1b"   # "gpipe" | "1f1b" | "interleaved"

    @property
    def bubble_fraction(self) -> float:
        """Fraction of time wasted in pipeline bubble."""
        p = self.n_stages
        m = self.n_microbatches
        return (p - 1) / (m + p - 1)

    @property
    def efficiency(self) -> float:
        return 1.0 - self.bubble_fraction

    def to_dict(self) -> dict:
        return {
            "n_microbatches": self.n_microbatches,
            "n_stages":       self.n_stages,
            "schedule_type":  self.schedule_type,
            "bubble_fraction": round(self.bubble_fraction, 4),
            "efficiency":      round(self.efficiency, 4),
        }


class PipelineEngine:
    """
    Pipeline parallel training engine.

    Splits a list of layers across pipeline stages and
    orchestrates micro-batch execution.

    Args:
        layers:         All transformer layers to split.
        world_cfg:      :class:`WorldConfig`.
        n_microbatches: Micro-batches to split global batch into.

    Example::

        engine = PipelineEngine(all_layers, world_cfg, n_microbatches=4)
        stage  = engine.get_stage(pp_rank=0)
        output = engine.forward_stage(stage, input_tensor)
    """

    def __init__(
        self,
        layers:         list[nn.Module],
        world_cfg:      WorldConfig,
        n_microbatches: int = 4,
    ) -> None:
        self.world_cfg      = world_cfg
        self.n_microbatches = n_microbatches
        self.pp_size        = world_cfg.pp_size
        self.stages         = self._partition_layers(layers)
        self.schedule       = PipelineSchedule(
            n_microbatches = n_microbatches,
            n_stages       = self.pp_size,
        )

    def _partition_layers(self, layers: list[nn.Module]) -> list[PipelineStage]:
        """Split layers evenly across pipeline stages."""
        n = len(layers)
        per_stage = max(1, n // self.pp_size)
        stages    = []
        for sid in range(self.pp_size):
            start = sid * per_stage
            end   = start + per_stage if sid < self.pp_size - 1 else n
            stage_layers = nn.ModuleList(layers[start:end])
            stages.append(PipelineStage(
                stage_id = sid,
                layers   = stage_layers,
                is_first = (sid == 0),
                is_last  = (sid == self.pp_size - 1),
            ))
        return stages

    def get_stage(self, pp_rank: int | None = None) -> PipelineStage:
        """Get the pipeline stage for a given pp_rank."""
        rank = pp_rank if pp_rank is not None else self.world_cfg.pp_rank
        return self.stages[min(rank, len(self.stages) - 1)]

    def forward_stage(
        self,
        stage: PipelineStage,
        x:     torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass for one stage on one micro-batch."""
        return stage.forward(x)

    def forward_microbatches(
        self,
        inputs:    list[torch.Tensor],
        pp_rank:   int = 0,
    ) -> list[torch.Tensor]:
        """
        Run simulated pipeline forward for all micro-batches on this stage.

        Args:
            inputs:  List of micro-batch tensors.
            pp_rank: This stage's pipeline rank.

        Returns:
            List of output tensors for each micro-batch.
        """
        stage   = self.get_stage(pp_rank)
        outputs = []
        for x in inputs:
            y = self.forward_stage(stage, x)
            outputs.append(y)
        return outputs

    def memory_per_stage(self, d_model: int, seq_len: int, batch_size: int) -> dict:
        """Estimate activation memory per pipeline stage."""
        # Each micro-batch stores activations for all layers in stage
        stage      = self.get_stage(0)
        n_layers   = len(stage.layers)
        activation_bytes = (
            batch_size // self.n_microbatches *
            seq_len * d_model * 4 * n_layers
        )
        return {
            "n_layers_per_stage": n_layers,
            "activation_mb":      round(activation_bytes / 1e6, 2),
            "bubble_efficiency":  round(self.schedule.efficiency, 3),
        }
