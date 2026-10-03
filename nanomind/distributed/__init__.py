"""NanoMind Distributed sub-package — Distributed Training infrastructure.

Implements the full distributed training stack:
  1. WorldConfig             — dp/tp/pp topology, ranks
  2. MockProcessGroup        — simulated comm groups for testing
  3. build_process_groups    — build dp/tp/pp groups
  4. DDPConfig               — bucket size, find_unused_params
  5. DataParallelWrapper     — DDP with bucket-based allreduce
  6. DDPStats                — allreduce calls, comm bytes
  7. ZeROStats               — per-stage memory savings (ZeRO-1/2/3)
  8. ColumnParallelLinear    — split output across TP ranks
  9. RowParallelLinear       — split input across TP ranks
  10. TensorParallelMLP      — column-parallel fc1, row-parallel fc2
  11. PipelineStage          — one PP stage with its layers
  12. PipelineEngine         — partition layers, run micro-batches
  13. PipelineSchedule       — bubble fraction, efficiency
  14. CheckpointingConfig    — ratio, offload, recompute_attention
  15. CheckpointedLayer      — torch.utils.checkpoint wrapper
  16. SelectiveCheckpointing — apply to subset of layers
  17. estimate_activation_memory — per-layer memory breakdown
  18. AMPConfig              — dtype, scale, backoff
  19. LossScaler             — fp16 dynamic loss scaling
  20. MixedPrecisionTrainer  — backward, step, memory_report

Primary exports:
    - :class:`WorldConfig`              — 3D parallelism topology
    - :class:`DataParallelWrapper`      — DDP gradient averaging
    - :class:`ZeROStats`                — memory savings analysis
    - :class:`ColumnParallelLinear`     — Megatron-LM column-parallel
    - :class:`RowParallelLinear`        — Megatron-LM row-parallel
    - :class:`TensorParallelMLP`        — TP MLP block
    - :class:`PipelineEngine`           — PP layer partitioning
    - :class:`PipelineSchedule`         — bubble fraction
    - :class:`CheckpointedLayer`        — activation recomputation
    - :class:`SelectiveCheckpointing`   — apply checkpointing
    - :func:`estimate_activation_memory` — memory breakdown
    - :class:`AMPConfig`                — AMP configuration
    - :class:`LossScaler`               — fp16 dynamic scaling
    - :class:`MixedPrecisionTrainer`    — full AMP training wrapper
"""

from nanomind.distributed.world import WorldConfig, MockProcessGroup, build_process_groups
from nanomind.distributed.data_parallel import (
    DDPConfig, DDPStats, DataParallelWrapper, ZeROStats,
)
from nanomind.distributed.tensor_parallel import (
    ColumnParallelLinear, RowParallelLinear, TensorParallelMLP,
)
from nanomind.distributed.pipeline import (
    PipelineStage, PipelineEngine, PipelineSchedule, MicroBatch,
)
from nanomind.distributed.checkpointing import (
    CheckpointingConfig, CheckpointedLayer, SelectiveCheckpointing,
    estimate_activation_memory,
)
from nanomind.distributed.mixed_precision import (
    AMPConfig, LossScaler, MixedPrecisionTrainer,
)

__all__ = [
    "WorldConfig", "MockProcessGroup", "build_process_groups",
    "DDPConfig", "DDPStats", "DataParallelWrapper", "ZeROStats",
    "ColumnParallelLinear", "RowParallelLinear", "TensorParallelMLP",
    "PipelineStage", "PipelineEngine", "PipelineSchedule", "MicroBatch",
    "CheckpointingConfig", "CheckpointedLayer", "SelectiveCheckpointing",
    "estimate_activation_memory",
    "AMPConfig", "LossScaler", "MixedPrecisionTrainer",
]
