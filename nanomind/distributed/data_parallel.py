"""
nanomind/distributed/data_parallel.py — Data Parallel training (DDP).

## Data Parallelism

Each GPU holds a complete copy of the model.
Each iteration:
  1. Split mini-batch across GPUs: B/N samples per GPU
  2. Forward pass on each GPU independently
  3. Backward pass → local gradients
  4. AllReduce: average gradients across all GPUs
  5. Apply same gradient update on all GPUs → models stay in sync

## Gradient Communication

Naive DDP: wait for backward, then AllReduce all gradients
  → idle GPU time during AllReduce

Overlapped DDP (PyTorch DistributedDataParallel):
  → Bucket gradients, AllReduce as soon as bucket is full
  → Overlaps backward computation with gradient communication

## ZeRO (Zero Redundancy Optimizer)

Each GPU stores only 1/N of the optimizer state:
  ZeRO-1: Partition optimizer states (8× memory reduction)
  ZeRO-2: + Partition gradients     (8× + gradient memory)
  ZeRO-3: + Partition parameters    (full 3D sharding)

With ZeRO-3, you can train 1T parameter models on 512 GPUs!

NanoMind implements simulated DDP with:
  - Gradient averaging (AllReduce simulation)
  - Bucket-based gradient communication
  - ZeRO-1 optimizer state sharding stats

Reference:
  Rajbhandari et al. (2020) "ZeRO: Memory Optimizations Toward Training
  Trillion Parameter Models" https://arxiv.org/abs/1910.02054
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from nanomind.distributed.world import WorldConfig


@dataclass
class DDPConfig:
    """Configuration for data parallel training."""
    bucket_size_mb: float = 25.0      # Gradient bucket size in MB
    find_unused_params: bool = False
    gradient_as_bucket_view: bool = True
    sync_batch_norm: bool = False


@dataclass
class DDPStats:
    """Statistics from DDP gradient communication."""
    n_allreduce_calls: int   = 0
    total_comm_bytes:  int   = 0
    n_buckets:         int   = 0

    @property
    def total_comm_mb(self) -> float:
        return self.total_comm_bytes / 1e6

    def to_dict(self) -> dict:
        return {
            "n_allreduce_calls": self.n_allreduce_calls,
            "total_comm_mb":     round(self.total_comm_mb, 2),
            "n_buckets":         self.n_buckets,
        }


class DataParallelWrapper:
    """
    Simulated DDP wrapper for a PyTorch module.

    In production, use torch.nn.parallel.DistributedDataParallel.
    This wrapper simulates the gradient averaging logic.

    Args:
        module:     Model to wrap.
        world_cfg:  :class:`WorldConfig`.
        ddp_cfg:    :class:`DDPConfig`.

    Example::

        wrapper = DataParallelWrapper(model, world_cfg, ddp_cfg)
        # Simulate forward + backward
        loss.backward()
        wrapper.finish_gradient_synchronization()
        # Gradients are now averaged across simulated world_size GPUs
    """

    def __init__(
        self,
        module:    nn.Module,
        world_cfg: WorldConfig,
        ddp_cfg:   DDPConfig | None = None,
    ) -> None:
        self.module    = module
        self.world_cfg = world_cfg
        self.ddp_cfg   = ddp_cfg or DDPConfig()
        self.stats     = DDPStats()
        self._buckets  = self._build_buckets()

    def _build_buckets(self) -> list[list[nn.Parameter]]:
        """Group parameters into gradient buckets."""
        bucket_bytes = self.ddp_cfg.bucket_size_mb * 1e6
        buckets: list[list[nn.Parameter]] = [[]]
        current_bytes = 0
        for p in self.module.parameters():
            if p.requires_grad:
                p_bytes = p.numel() * p.element_size()
                if current_bytes + p_bytes > bucket_bytes and buckets[-1]:
                    buckets.append([])
                    current_bytes = 0
                buckets[-1].append(p)
                current_bytes += p_bytes
        return [b for b in buckets if b]

    def allreduce_gradients(self) -> None:
        """
        Simulate AllReduce: average gradients across world_size.

        In real DDP: torch.distributed.all_reduce(grad, op=dist.ReduceOp.AVG)
        Simulation: divide gradients by world_size.
        """
        for bucket in self._buckets:
            grads = [p.grad for p in bucket if p.grad is not None]
            if grads:
                # Simulate averaging across world_size GPUs
                for g in grads:
                    g.div_(self.world_cfg.dp_size)
                # Track communication stats
                comm_bytes = sum(g.numel() * g.element_size() for g in grads)
                self.stats.n_allreduce_calls += 1
                self.stats.total_comm_bytes  += comm_bytes
        self.stats.n_buckets = len(self._buckets)

    def finish_gradient_synchronization(self) -> None:
        """Called after backward pass to synchronize gradients."""
        self.allreduce_gradients()

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.module.parameters()
                   if p.requires_grad)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ZeROStats:
    """
    Statistics for ZeRO optimizer memory savings.

    Shows theoretical memory reduction for each ZeRO stage.
    """

    def __init__(self, model: nn.Module, world_size: int, mixed_precision: bool = True) -> None:
        self.n_params    = sum(p.numel() for p in model.parameters())
        self.world_size  = world_size
        self.bytes_per_param_fp32 = 4
        self.bytes_per_param_fp16 = 2 if mixed_precision else 4

    def baseline_bytes(self) -> int:
        """Memory per GPU without any optimization (Adam, fp32)."""
        # Params (fp32) + gradients (fp32) + Adam states (2×fp32 momentum)
        return self.n_params * (4 + 4 + 8)  # 16 bytes per param

    def zero1_bytes(self) -> int:
        """ZeRO-1: shard optimizer states."""
        params_grads = self.n_params * 8   # params + grads (full)
        optim_states = self.n_params * 8 // self.world_size  # sharded
        return params_grads + optim_states

    def zero2_bytes(self) -> int:
        """ZeRO-2: shard optimizer states + gradients."""
        params = self.n_params * 4   # full params
        grads_optim = self.n_params * (4 + 8) // self.world_size
        return params + grads_optim

    def zero3_bytes(self) -> int:
        """ZeRO-3: shard params + gradients + optimizer states."""
        return self.n_params * 16 // self.world_size

    def summary(self) -> dict:
        base = self.baseline_bytes()
        return {
            "n_params":       self.n_params,
            "baseline_gb":    round(base / 1e9, 3),
            "zero1_gb":       round(self.zero1_bytes() / 1e9, 3),
            "zero2_gb":       round(self.zero2_bytes() / 1e9, 3),
            "zero3_gb":       round(self.zero3_bytes() / 1e9, 3),
            "zero3_speedup":  round(base / max(self.zero3_bytes(), 1), 1),
        }
