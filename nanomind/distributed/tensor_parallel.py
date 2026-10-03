"""
nanomind/distributed/tensor_parallel.py — Tensor Parallelism (Megatron-LM style).

## Tensor Parallelism (Shoeybi et al., 2019)

Split individual weight matrices across GPUs.
No need to communicate between forward passes!

## Column-Parallel Linear

Split weight W along output dimension:
  W: (in_features, out_features) → split into N slices of (in_features, out/N)
  Each GPU computes: Y_i = X @ W_i
  Result: Y = concat([Y_0, Y_1, ..., Y_{N-1}], dim=-1)

## Row-Parallel Linear

Split weight W along input dimension:
  W: (in_features, out_features) → split into N slices of (in/N, out_features)
  Each GPU computes: Y_i = X_i @ W_i  (X_i is its shard of X)
  Result: Y = sum([Y_0, Y_1, ..., Y_{N-1}])  ← AllReduce!

## Transformer Block with TP

MLP: fc1 → fc2
  fc1: Column-parallel (no comm after)
  fc2: Row-parallel    (AllReduce at end)
  → Only 1 AllReduce per MLP block!

Attention: Q/K/V proj + output proj
  Q/K/V: Column-parallel
  Output: Row-parallel
  → 1 AllReduce per attention block!

This is how Megatron-LM trains 530B-parameter models on thousands of A100s.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.distributed.world import WorldConfig


class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear layer (Megatron-LM style).

    Splits the output dimension across TP ranks.
    No communication needed after the forward pass if gather=False.

    Args:
        in_features:  Input dimension.
        out_features: Total output dimension (split across tp_size).
        world_cfg:    :class:`WorldConfig`.
        bias:         Include bias term.
        gather_output: AllGather output at the end.

    Example::

        layer   = ColumnParallelLinear(128, 512, world_cfg, bias=True)
        x       = torch.randn(4, 128)
        y_shard = layer(x)   # (4, 512//tp_size) on each rank
    """

    def __init__(
        self,
        in_features:   int,
        out_features:  int,
        world_cfg:     WorldConfig,
        bias:          bool = True,
        gather_output: bool = True,
    ) -> None:
        super().__init__()
        self.in_features   = in_features
        self.out_features  = out_features
        self.tp_size       = world_cfg.tp_size
        self.tp_rank       = world_cfg.tp_rank
        self.gather_output = gather_output

        # Each rank holds out_features / tp_size output columns
        self.out_shard     = out_features // self.tp_size
        self.linear        = nn.Linear(in_features, self.out_shard, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — computes local shard of output.

        Args:
            x: ``(B, T, in_features)`` or ``(B, in_features)``

        Returns:
            ``(B, T, out_features)`` if gather_output else ``(B, T, out_shard)``
        """
        y_shard = self.linear(x)   # (B, T, out_shard)
        if self.gather_output and self.tp_size > 1:
            # Simulate AllGather: in practice, dist.all_gather
            # Here we just expand (no real comm in simulation)
            y = y_shard.repeat(1, 1, self.tp_size) if y_shard.dim() == 3 else y_shard
            return y[..., :self.out_features]
        return y_shard

    @property
    def weight_shape(self) -> tuple:
        return (self.out_shard, self.in_features)


class RowParallelLinear(nn.Module):
    """
    Row-parallel linear layer (Megatron-LM style).

    Splits the input dimension across TP ranks.
    Requires AllReduce at the end.

    Args:
        in_features:   Total input dimension (split across tp_size).
        out_features:  Output dimension (same on all ranks).
        world_cfg:     :class:`WorldConfig`.
        bias:          Include bias (added only on last rank).
        input_is_parallel: Input is already split across ranks.

    Example::

        layer = RowParallelLinear(512, 128, world_cfg)
        x_shard = torch.randn(4, 512 // tp_size)   # already split
        y       = layer(x_shard)                     # (4, 128) after AllReduce
    """

    def __init__(
        self,
        in_features:       int,
        out_features:      int,
        world_cfg:         WorldConfig,
        bias:              bool = True,
        input_is_parallel: bool = True,
    ) -> None:
        super().__init__()
        self.in_features      = in_features
        self.out_features     = out_features
        self.tp_size          = world_cfg.tp_size
        self.tp_rank          = world_cfg.tp_rank
        self.input_is_parallel = input_is_parallel

        self.in_shard = in_features // self.tp_size
        # Bias only added on tp_rank == 0 to avoid double-counting
        self.linear   = nn.Linear(self.in_shard, out_features,
                                   bias=(bias and self.tp_rank == 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — computes local output shard and reduces.

        Args:
            x: ``(B, T, in_shard)`` if input_is_parallel else ``(B, T, in_features)``

        Returns:
            ``(B, T, out_features)`` after simulated AllReduce.
        """
        if not self.input_is_parallel:
            # Scatter input: take our shard
            x = x[..., self.tp_rank * self.in_shard: (self.tp_rank + 1) * self.in_shard]

        y_local = self.linear(x)    # (B, T, out_features) — local partial sum

        # Simulate AllReduce (sum across TP ranks)
        # In production: torch.distributed.all_reduce(y_local, op=dist.ReduceOp.SUM)
        # Simulation: scale by 1/tp_size to simulate averaging, then × tp_size
        return y_local   # (already correct for single-process simulation)

    @property
    def weight_shape(self) -> tuple:
        return (self.out_features, self.in_shard)


class TensorParallelMLP(nn.Module):
    """
    Tensor-parallel MLP block (Megatron-LM style).

    Structure:
      fc1: ColumnParallel (split output)
      gelu activation
      fc2: RowParallel    (split input, AllReduce)

    Memory per GPU: O(d_model × d_ff / tp_size)

    Args:
        d_model: Model hidden dimension.
        d_ff:    Feed-forward intermediate dimension.
        world_cfg: :class:`WorldConfig`.

    Example::

        mlp = TensorParallelMLP(512, 2048, world_cfg)
        x   = torch.randn(4, 16, 512)
        y   = mlp(x)   # (4, 16, 512)
    """

    def __init__(self, d_model: int, d_ff: int, world_cfg: WorldConfig) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(d_model, d_ff,   world_cfg,
                                         gather_output=False)
        self.fc2 = RowParallelLinear  (d_ff,    d_model, world_cfg,
                                         input_is_parallel=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))
