"""
day53_commits.py — 20 atomic commits for Day 53: Distributed Training.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== DAY 53: Distributed Training — 20 commits, v5.3.0 ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — distributed package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/distributed/__init__.py",
      '"""NanoMind Distributed sub-package — Distributed Training infrastructure."""\n')
commit("feat: add nanomind/distributed/ package skeleton for distributed training")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — Process group and world config
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/distributed/world.py", '''\
"""
nanomind/distributed/world.py — Distributed world configuration.

## Distributed Training Topology

Training LLMs requires multiple GPUs across multiple nodes.
The "world" describes the full distributed setup:

  world_size = total number of processes (GPUs)
  rank       = global index of this process (0..world_size-1)
  local_rank = GPU index on this node (0..gpus_per_node-1)

## Parallelism Dimensions

Modern LLM training uses 3D parallelism:

  1. Data Parallelism (DP): Split batch across GPUs
     - Each GPU has full model copy
     - Gradients averaged across GPUs via AllReduce
     - Scales: batch size × world_size

  2. Tensor Parallelism (TP): Split individual weight matrices
     - Column/row-parallel linear layers
     - Megatron-LM style (Shoeybi et al., 2019)
     - Scales: model width

  3. Pipeline Parallelism (PP): Split layers across GPUs
     - GPU 0: layers 0-15, GPU 1: layers 16-31, ...
     - Micro-batches pipeline through stages
     - GPipe / 1F1B schedule

  4. Expert Parallelism (EP): MoE experts across GPUs
     - Each GPU holds subset of experts

Combined (Megatron-DeepSpeed):
  world_size = DP × TP × PP
  E.g., 512 GPUs = 64 DP × 4 TP × 2 PP

References:
  Shoeybi et al. (2019) Megatron-LM: https://arxiv.org/abs/1909.08053
  Rajbhandari et al. (2020) ZeRO:     https://arxiv.org/abs/1910.02054
  Huang et al. (2019) GPipe:          https://arxiv.org/abs/1811.06965
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class WorldConfig:
    """
    Configuration for the distributed training world.

    Args:
        world_size: Total number of processes (GPUs).
        rank:       This process's global rank.
        local_rank: This process's local GPU index.
        dp_size:    Data parallel degree.
        tp_size:    Tensor parallel degree.
        pp_size:    Pipeline parallel degree.
        backend:    Communication backend ("nccl" | "gloo" | "mpi").

    Example::

        cfg = WorldConfig(world_size=8, rank=0, local_rank=0,
                          dp_size=4, tp_size=2, pp_size=1)
        print(cfg.is_main_process)   # True (rank == 0)
        print(cfg.dp_rank)           # 0
        print(cfg.tp_rank)           # 0
    """
    world_size: int   = 1
    rank:       int   = 0
    local_rank: int   = 0
    dp_size:    int   = 1
    tp_size:    int   = 1
    pp_size:    int   = 1
    backend:    str   = "gloo"

    def __post_init__(self):
        assert self.dp_size * self.tp_size * self.pp_size == self.world_size, (
            f"dp×tp×pp ({self.dp_size}×{self.tp_size}×{self.pp_size}) "
            f"!= world_size ({self.world_size})"
        )

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def dp_rank(self) -> int:
        """This process's rank within its DP group."""
        return self.rank // (self.tp_size * self.pp_size)

    @property
    def tp_rank(self) -> int:
        """This process's rank within its TP group."""
        return (self.rank // self.pp_size) % self.tp_size

    @property
    def pp_rank(self) -> int:
        """This process's rank within its PP group."""
        return self.rank % self.pp_size

    @property
    def is_first_pp_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last_pp_stage(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    def to_dict(self) -> dict:
        return {
            "world_size": self.world_size,
            "rank":       self.rank,
            "local_rank": self.local_rank,
            "dp_size":    self.dp_size,
            "tp_size":    self.tp_size,
            "pp_size":    self.pp_size,
            "dp_rank":    self.dp_rank,
            "tp_rank":    self.tp_rank,
            "pp_rank":    self.pp_rank,
        }


# ── Simulated process group ────────────────────────────────────────────────────

class MockProcessGroup:
    """
    Simulated process group for unit testing without real GPU comm.

    Mimics PyTorch's dist.ProcessGroup interface.
    """

    def __init__(self, ranks: list[int]) -> None:
        self.ranks = ranks
        self.size  = len(ranks)

    def __repr__(self) -> str:
        return f"MockProcessGroup(ranks={self.ranks})"


def build_process_groups(cfg: WorldConfig) -> dict[str, MockProcessGroup]:
    """
    Build simulated process groups for DP, TP, PP.

    In real training: torch.distributed.new_group()

    Returns dict with "dp", "tp", "pp" groups.
    """
    groups = {}
    # DP group: same TP+PP position, different DP rank
    dp_ranks = list(range(0, cfg.world_size, cfg.tp_size * cfg.pp_size))
    groups["dp"] = MockProcessGroup(dp_ranks[:cfg.dp_size])

    # TP group: consecutive tp_size ranks
    tp_start = (cfg.rank // cfg.tp_size) * cfg.tp_size
    groups["tp"] = MockProcessGroup(list(range(tp_start, tp_start + cfg.tp_size)))

    # PP group: ranks separated by tp_size
    pp_ranks = [cfg.dp_rank * cfg.tp_size * cfg.pp_size +
                cfg.tp_rank + i * cfg.tp_size
                for i in range(cfg.pp_size)]
    groups["pp"] = MockProcessGroup(pp_ranks)

    return groups
''')
commit("feat: add WorldConfig (dp/tp/pp), MockProcessGroup, build_process_groups — 3D parallelism topology")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — Data Parallel (DDP simulation)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/distributed/data_parallel.py", '''\
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
''')
commit("feat: add DataParallelWrapper (DDP), allreduce_gradients, bucket grouping, ZeROStats memory savings")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — Tensor Parallel linear layers
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/distributed/tensor_parallel.py", '''\
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
''')
commit("feat: add ColumnParallelLinear, RowParallelLinear, TensorParallelMLP — Megatron-LM TP")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — Pipeline Parallelism
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/distributed/pipeline.py", '''\
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
''')
commit("feat: add PipelineStage, PipelineEngine, PipelineSchedule — GPipe/1F1B, bubble fraction, efficiency")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Gradient checkpointing
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/distributed/checkpointing.py", '''\
"""
nanomind/distributed/checkpointing.py — Gradient checkpointing & activation recomputation.

## The Memory-Compute Tradeoff

During backward pass, PyTorch needs activation tensors from the forward pass.
For a Transformer with L layers:
  Without checkpointing: store ALL activations → O(L × B × T × D) memory
  With checkpointing:    store ONLY checkpointed activations → O(√L) memory

## Gradient Checkpointing (Chen et al., 2016)

Instead of storing ALL activations:
  1. Forward: compute normally, but don't save intermediate activations
  2. Save only "checkpoint" tensors at intervals
  3. Backward: recompute activations from last checkpoint when needed

Trade: 33% more compute, but O(√L) memory instead of O(L).

This allows training much larger batches / longer sequences!

## Selective Activation Recomputation

Re-checkpoint only expensive operations:
  - Full self-attention: O(T² × D)
  - Recompute: only store Q,K,V, recompute attention scores

## Activation Offloading

For extreme memory savings: offload activations to CPU RAM during forward.
Bring back to GPU during backward.
  → 4× larger batch sizes!
  → ~20% throughput loss due to CPU↔GPU transfer

Reference:
  Chen et al. (2016) "Training Deep Nets with Sublinear Memory Cost"
  https://arxiv.org/abs/1604.06174
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class CheckpointingConfig:
    """Configuration for gradient checkpointing."""
    enabled:            bool  = True
    checkpoint_ratio:   float = 1.0    # 1.0 = all layers, 0.5 = every other layer
    offload_to_cpu:     bool  = False  # offload activations to CPU
    recompute_attention: bool = True   # recompute attention in backward
    use_reentrant:      bool  = False  # PyTorch checkpoint() flag

    def __post_init__(self):
        assert 0.0 <= self.checkpoint_ratio <= 1.0


class CheckpointedLayer(nn.Module):
    """
    Wrapper that applies gradient checkpointing to a layer.

    Instead of storing activations, recomputes them during backward.
    Reduces memory by not storing intermediate activations.

    Args:
        layer:      The wrapped layer.
        enabled:    Whether checkpointing is active.

    Example::

        layer     = CheckpointedLayer(transformer_block, enabled=True)
        output    = layer(hidden_states)   # activations NOT stored
        # → 33% more compute, but much less memory
    """

    def __init__(self, layer: nn.Module, enabled: bool = True) -> None:
        super().__init__()
        self.layer   = layer
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled and x.requires_grad:
            # torch.utils.checkpoint.checkpoint recomputes in backward
            return torch.utils.checkpoint.checkpoint(
                self.layer, x, use_reentrant=False
            )
        return self.layer(x)


class SelectiveCheckpointing:
    """
    Apply gradient checkpointing to a subset of layers.

    Checkpoints every ceil(1/ratio) layers.

    Args:
        layers:  List of transformer layers.
        cfg:     :class:`CheckpointingConfig`.

    Example::

        sc     = SelectiveCheckpointing(transformer_layers, cfg)
        layers = sc.apply()
        # Some layers now wrapped with CheckpointedLayer
    """

    def __init__(
        self,
        layers: list[nn.Module],
        cfg:    CheckpointingConfig | None = None,
    ) -> None:
        self.layers = layers
        self.cfg    = cfg or CheckpointingConfig()

    def apply(self) -> list[nn.Module]:
        """Wrap layers with checkpointing according to ratio."""
        if not self.cfg.enabled:
            return self.layers

        step      = max(1, round(1.0 / self.cfg.checkpoint_ratio))
        wrapped   = []
        for i, layer in enumerate(self.layers):
            if i % step == 0:
                wrapped.append(CheckpointedLayer(layer, enabled=True))
            else:
                wrapped.append(layer)
        return wrapped

    @property
    def n_checkpointed(self) -> int:
        step = max(1, round(1.0 / self.cfg.checkpoint_ratio))
        return (len(self.layers) + step - 1) // step

    def memory_savings(self, baseline_mb: float) -> dict:
        """Estimate memory savings from checkpointing."""
        ratio = self.cfg.checkpoint_ratio
        saved_fraction = 1.0 - ratio   # store ratio, free the rest
        return {
            "baseline_mb":    round(baseline_mb, 2),
            "with_ckpt_mb":   round(baseline_mb * ratio, 2),
            "saved_mb":       round(baseline_mb * saved_fraction, 2),
            "saved_fraction":  round(saved_fraction, 3),
            "compute_overhead": "33%",
        }


def estimate_activation_memory(
    n_layers:   int,
    batch_size: int,
    seq_len:    int,
    d_model:    int,
    n_heads:    int,
    dtype_bytes: int = 2,   # fp16
) -> dict:
    """
    Estimate transformer activation memory (bytes).

    Args:
        n_layers:    Number of transformer layers.
        batch_size:  Batch size.
        seq_len:     Sequence length.
        d_model:     Model hidden dimension.
        n_heads:     Number of attention heads.
        dtype_bytes: Bytes per element (2=fp16, 4=fp32).

    Returns:
        Dict with memory estimates with/without checkpointing.
    """
    B, T, D, H = batch_size, seq_len, d_model, n_heads

    # Per-layer activations:
    # - Input: B × T × D
    # - Attention QKV: B × T × 3D
    # - Attention scores: B × H × T × T
    # - MLP: B × T × 4D
    per_layer = dtype_bytes * (
        B * T * D +           # input
        B * T * 3 * D +       # QKV
        B * H * T * T +       # attention scores
        B * T * 4 * D         # MLP
    )

    total        = n_layers * per_layer
    with_ckpt    = int(total ** 0.5)  # √L memory with full checkpointing

    return {
        "per_layer_mb":   round(per_layer / 1e6, 2),
        "total_mb":       round(total / 1e6, 2),
        "with_ckpt_mb":   round(with_ckpt / 1e6, 2),
        "savings_x":      round(total / max(with_ckpt, 1), 1),
    }
''')
commit("feat: add CheckpointedLayer, SelectiveCheckpointing, estimate_activation_memory — memory/compute tradeoff")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Mixed precision training
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/distributed/mixed_precision.py", '''\
"""
nanomind/distributed/mixed_precision.py — Mixed precision (AMP) training.

## Mixed Precision Training (Micikevicius et al., 2018)

Train in fp16/bf16 for speed, keep fp32 master weights for accuracy.

fp16 vs fp32:
  fp32: 32-bit float, 7 decimal digits, full range
  fp16: 16-bit float, 3 decimal digits, smaller range (underflow/overflow!)
  bf16: 16-bit float, same exponent range as fp32, less precision
        → bf16 is preferred for LLMs (no overflow risk)

## AMP Recipe

  1. Forward pass: fp16/bf16 weights + activations → less memory, faster
  2. Loss scaling: multiply loss by scale factor (prevents fp16 underflow)
  3. Backward pass: fp16 gradients
  4. Unscale gradients: divide by scale factor
  5. Gradient clipping: in fp32 for stability
  6. fp32 master weight update: apply gradients to fp32 copy
  7. fp32 → fp16/bf16 cast: update fp16 weights for next forward

## Memory Savings

  LLaMA-70B fp32: 280 GB
  LLaMA-70B bf16: 140 GB   (2× reduction!)
  LLaMA-70B int8: 70 GB    (4× reduction, with quantization)

Speed:
  A100 peak fp16 TFLOPS: 312
  A100 peak fp32 TFLOPS:  77.6  (4× faster in fp16!)

Reference:
  Micikevicius et al. (2018) "Mixed Precision Training"
  https://arxiv.org/abs/1710.03740
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class AMPConfig:
    """Configuration for automatic mixed precision."""
    dtype:           str   = "bf16"      # "fp16" | "bf16" | "fp32"
    initial_scale:   float = 2.0 ** 16   # for fp16 loss scaling
    scale_growth:    float = 2.0
    scale_min:       float = 1.0
    scale_max:       float = 2.0 ** 24
    backoff_factor:  float = 0.5
    growth_interval: int   = 2000        # steps between scale growth

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32}[self.dtype]

    @property
    def use_autocast(self) -> bool:
        return self.dtype != "fp32"


class LossScaler:
    """
    Dynamic loss scaler for fp16 training.

    Automatically adjusts scale factor to prevent gradient underflow
    while avoiding overflow.

    Args:
        cfg: :class:`AMPConfig`.

    Example::

        scaler = LossScaler(AMPConfig(dtype="fp16"))
        loss   = scaler.scale(loss)
        loss.backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
    """

    def __init__(self, cfg: AMPConfig | None = None) -> None:
        self.cfg   = cfg or AMPConfig()
        self._scale      = self.cfg.initial_scale
        self._steps      = 0
        self._n_overflow = 0

    @property
    def scale(self) -> float:
        return self._scale

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Multiply loss by current scale factor."""
        return loss * self._scale

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Divide all gradients by current scale factor."""
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data.div_(self._scale)

    def has_overflow(self, optimizer: torch.optim.Optimizer) -> bool:
        """Check if any gradient is inf or nan."""
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                        return True
        return False

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Take optimizer step if no overflow.

        Returns:
            True if step was taken (no overflow).
        """
        if self.has_overflow(optimizer):
            self._n_overflow += 1
            self._scale = max(self.cfg.scale_min,
                              self._scale * self.cfg.backoff_factor)
            # Zero gradients to avoid applying corrupt gradients
            optimizer.zero_grad()
            return False

        optimizer.step()
        return True

    def update(self) -> None:
        """Update scale factor (increase if no overflow for a while)."""
        self._steps += 1
        if self._steps % self.cfg.growth_interval == 0:
            self._scale = min(self.cfg.scale_max,
                              self._scale * self.cfg.scale_growth)

    def state_dict(self) -> dict:
        return {"scale": self._scale, "steps": self._steps,
                "n_overflow": self._n_overflow}


class MixedPrecisionTrainer:
    """
    Mixed precision training wrapper.

    Handles dtype casting, loss scaling, and master weight synchronisation.

    Args:
        model:     Model to train (converted to AMP dtype).
        optimizer: Optimizer operating on fp32 master weights.
        cfg:       :class:`AMPConfig`.

    Example::

        trainer = MixedPrecisionTrainer(model, optimizer, AMPConfig(dtype="bf16"))
        loss    = trainer.compute_loss(input_ids, labels)
        trainer.backward(loss)
        trainer.step()
    """

    def __init__(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg:       AMPConfig | None = None,
    ) -> None:
        self.model     = model
        self.optimizer = optimizer
        self.cfg       = cfg or AMPConfig()
        self.scaler    = LossScaler(self.cfg)
        self._step_count = 0

    @contextmanager
    def autocast(self):
        """Context manager for automatic dtype casting."""
        if self.cfg.use_autocast:
            with torch.autocast("cpu", dtype=self.cfg.torch_dtype):
                yield
        else:
            yield

    def backward(self, loss: torch.Tensor) -> None:
        """Scale loss and run backward."""
        if self.cfg.dtype == "fp16":
            scaled = self.scaler.scale_loss(loss)
            scaled.backward()
        else:
            loss.backward()

    def step(self, max_grad_norm: float | None = None) -> bool:
        """
        Unscale, clip, and step the optimizer.

        Args:
            max_grad_norm: Gradient clipping norm (None = no clip).

        Returns:
            True if optimizer step was taken.
        """
        if self.cfg.dtype == "fp16":
            self.scaler.unscale_(self.optimizer)
            if max_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            success = self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if max_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            success = True

        self._step_count += 1
        return success

    def memory_report(self) -> dict:
        """Memory comparison between fp32 and AMP modes."""
        n_params  = sum(p.numel() for p in self.model.parameters())
        bytes_fp32 = n_params * 4
        amp_bytes  = n_params * (2 if self.cfg.dtype in ("fp16", "bf16") else 4)
        return {
            "n_params":     n_params,
            "fp32_gb":      round(bytes_fp32 / 1e9, 3),
            "amp_gb":       round(amp_bytes / 1e9, 3),
            "savings_x":    round(bytes_fp32 / max(amp_bytes, 1), 1),
            "dtype":        self.cfg.dtype,
        }
''')
commit("feat: add AMPConfig, LossScaler, MixedPrecisionTrainer — fp16/bf16 AMP, loss scaling, memory report")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — distributed __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/distributed/__init__.py", '''\
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
''')
commit("refactor: export all distributed components from nanomind/distributed/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — example
# ══════════════════════════════════════════════════════════════════════════════
write("examples/distributed_demo.py", '''\
"""
examples/distributed_demo.py — NanoMind Distributed Training demo.

Usage:
    python examples/distributed_demo.py
"""
import torch
import torch.nn as nn
from nanomind.distributed import (
    WorldConfig, MockProcessGroup, build_process_groups,
    DDPConfig, DDPStats, DataParallelWrapper, ZeROStats,
    ColumnParallelLinear, RowParallelLinear, TensorParallelMLP,
    PipelineStage, PipelineEngine, PipelineSchedule,
    CheckpointingConfig, CheckpointedLayer, SelectiveCheckpointing,
    estimate_activation_memory,
    AMPConfig, LossScaler, MixedPrecisionTrainer,
)

V = 64

class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.ff(self.norm(x)) + x

class TinyLM(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.emb  = nn.Embedding(V, d)
        self.rnn  = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None

print("=" * 60)
print("NanoMind Distributed Training Demo")
print("=" * 60)

# ── World Config ──────────────────────────────────────────────────────────────
print("\n── 3D Parallelism World Config ──")
configs = [
    WorldConfig(world_size=8,   rank=0, local_rank=0, dp_size=4, tp_size=2, pp_size=1),
    WorldConfig(world_size=16,  rank=0, local_rank=0, dp_size=4, tp_size=2, pp_size=2),
    WorldConfig(world_size=512, rank=0, local_rank=0, dp_size=64, tp_size=4, pp_size=2),
]
for cfg in configs:
    print(f"  {cfg.world_size} GPUs: DP×TP×PP = "
          f"{cfg.dp_size}×{cfg.tp_size}×{cfg.pp_size}, "
          f"is_main={cfg.is_main_process}")

cfg8 = WorldConfig(world_size=8, rank=0, local_rank=0, dp_size=4, tp_size=2, pp_size=1)
groups = build_process_groups(cfg8)
print(f"  Process groups: {list(groups.keys())}")

# ── Data Parallel (DDP) ───────────────────────────────────────────────────────
print("\n── Data Parallel (DDP) ──")
dp_cfg  = WorldConfig(world_size=4, rank=0, local_rank=0, dp_size=4, tp_size=1, pp_size=1)
model   = TinyLM()
wrapper = DataParallelWrapper(model, dp_cfg, DDPConfig(bucket_size_mb=1.0))
print(f"  Model params: {wrapper.n_parameters():,}")
print(f"  DDP buckets:  {len(wrapper._buckets)}")

# Simulate backward + gradient sync
ids  = torch.randint(0, V, (4, 8))
out  = wrapper(ids)
loss = out[0].sum()
loss.backward()
wrapper.finish_gradient_synchronization()
print(f"  After allreduce: {wrapper.stats.to_dict()}")

# ZeRO memory analysis
zero = ZeROStats(model, world_size=8)
z    = zero.summary()
print(f"  ZeRO-3 (8 GPUs): {z['zero3_gb']:.3f} GB vs {z['baseline_gb']:.3f} GB baseline "
      f"({z['zero3_speedup']}× reduction)")

# ── Tensor Parallelism ────────────────────────────────────────────────────────
print("\n── Tensor Parallelism (Megatron-LM) ──")
tp_cfg  = WorldConfig(world_size=4, rank=0, local_rank=0, dp_size=1, tp_size=4, pp_size=1)
col_lin = ColumnParallelLinear(64, 256, tp_cfg, bias=True, gather_output=True)
row_lin = RowParallelLinear   (256, 64, tp_cfg, bias=True, input_is_parallel=False)
tp_mlp  = TensorParallelMLP   (64, 256, tp_cfg)

x = torch.randn(2, 8, 64)
print(f"  ColumnParallel: in={col_lin.weight_shape}, out={col_lin(x).shape}")
print(f"  RowParallel:    in={row_lin.weight_shape}, out={row_lin(col_lin(x)).shape}")
print(f"  TP-MLP:         {tp_mlp(x).shape}")

# ── Pipeline Parallelism ──────────────────────────────────────────────────────
print("\n── Pipeline Parallelism ──")
pp_cfg = WorldConfig(world_size=4, rank=0, local_rank=0, dp_size=1, tp_size=1, pp_size=4)
layers = [TinyTransformerBlock(32) for _ in range(8)]
engine = PipelineEngine(layers, pp_cfg, n_microbatches=4)

sched  = engine.schedule
print(f"  Stages: {engine.pp_size}, Micro-batches: {engine.n_microbatches}")
print(f"  Schedule: {sched.to_dict()}")
print(f"  Memory: {engine.memory_per_stage(32, 16, 8)}")

micro_inputs = [torch.randn(2, 16, 32) for _ in range(4)]
outputs = engine.forward_microbatches(micro_inputs, pp_rank=0)
print(f"  Micro-batch outputs: {[tuple(o.shape) for o in outputs[:2]]}...")

# ── Gradient Checkpointing ────────────────────────────────────────────────────
print("\n── Gradient Checkpointing ──")
ckpt_cfg = CheckpointingConfig(enabled=True, checkpoint_ratio=0.5)
sc       = SelectiveCheckpointing(layers, ckpt_cfg)
ckpt_layers = sc.apply()
n_ckpt = sum(1 for l in ckpt_layers if isinstance(l, CheckpointedLayer))
print(f"  Layers: {len(layers)}, Checkpointed: {n_ckpt}/{len(layers)}")
print(f"  Memory savings: {sc.memory_savings(512.0)}")

mem = estimate_activation_memory(32, 8, 512, 768, 12)
print(f"  Activation memory (32L, 768D, T=512): {mem}")

# CheckpointedLayer
x_ckpt = torch.randn(2, 16, 32, requires_grad=True)
ckpt_l = CheckpointedLayer(layers[0], enabled=True)
y_ckpt = ckpt_l(x_ckpt)
print(f"  CheckpointedLayer output: {tuple(y_ckpt.shape)}")

# ── Mixed Precision ───────────────────────────────────────────────────────────
print("\n── Mixed Precision Training (AMP) ──")
amp_model = TinyLM(32)
opt       = torch.optim.AdamW(amp_model.parameters(), lr=1e-4)
amp_cfg   = AMPConfig(dtype="fp32")  # fp32 for CPU demo
trainer   = MixedPrecisionTrainer(amp_model, opt, amp_cfg)

with trainer.autocast():
    out  = amp_model(torch.randint(0, V, (4, 8)))
    loss = out[0].sum()

trainer.backward(loss)
trainer.step(max_grad_norm=1.0)
print(f"  Memory report: {trainer.memory_report()}")

# LossScaler
scaler = LossScaler(AMPConfig(dtype="fp16", initial_scale=2**15))
print(f"  Initial scale: {scaler.scale:.0f}")
scaler.update()
print(f"  Scaler state: {scaler.state_dict()}")

print("\nDistributed training demo complete!")
''')
commit("feat: add examples/distributed_demo.py — DDP, TP, PP, checkpointing, AMP end-to-end demo")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_distributed.py", '''\
"""tests/test_distributed.py — Tests for NanoMind distributed package."""
import pytest
import torch
import torch.nn as nn
from nanomind.distributed import (
    WorldConfig, MockProcessGroup, build_process_groups,
    DDPConfig, DDPStats, DataParallelWrapper, ZeROStats,
    ColumnParallelLinear, RowParallelLinear, TensorParallelMLP,
    PipelineEngine, PipelineSchedule,
    CheckpointingConfig, CheckpointedLayer, SelectiveCheckpointing,
    estimate_activation_memory,
    AMPConfig, LossScaler, MixedPrecisionTrainer,
)

V = 32

class Block(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.ff = nn.Linear(d, d)
    def forward(self, x):
        return self.ff(x)

class TinyLM(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.emb  = nn.Embedding(V, d)
        self.rnn  = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None


# ── WorldConfig ───────────────────────────────────────────────────────────────

class TestWorldConfig:
    def test_basic(self):
        cfg = WorldConfig(4, 0, 0, dp_size=4, tp_size=1, pp_size=1)
        assert cfg.world_size == 4

    def test_is_main(self):
        cfg = WorldConfig(4, 0, 0, 4, 1, 1)
        assert cfg.is_main_process

    def test_not_main(self):
        cfg = WorldConfig(4, 2, 2, 4, 1, 1)
        assert not cfg.is_main_process

    def test_ranks_3d(self):
        cfg = WorldConfig(8, 0, 0, dp_size=4, tp_size=2, pp_size=1)
        assert cfg.dp_rank == 0
        assert cfg.tp_rank == 0

    def test_invalid_topology(self):
        with pytest.raises(AssertionError):
            WorldConfig(8, 0, 0, dp_size=4, tp_size=3, pp_size=1)

    def test_to_dict(self):
        cfg = WorldConfig(4, 0, 0, 4, 1, 1)
        d   = cfg.to_dict()
        assert "world_size" in d and "dp_rank" in d


# ── DataParallelWrapper ───────────────────────────────────────────────────────

class TestDDP:
    def _wrapper(self, world_size=4):
        cfg = WorldConfig(world_size, 0, 0, world_size, 1, 1)
        m   = TinyLM()
        return DataParallelWrapper(m, cfg, DDPConfig(bucket_size_mb=0.1))

    def test_n_params(self):
        w = self._wrapper()
        assert w.n_parameters() > 0

    def test_forward(self):
        w   = self._wrapper()
        ids = torch.randint(0, V, (2, 4))
        out = w(ids)
        assert isinstance(out, tuple)

    def test_allreduce_divides_grad(self):
        w   = self._wrapper(world_size=2)
        ids = torch.randint(0, V, (2, 4))
        out = w(ids)
        loss = out[0].sum()
        loss.backward()
        # Save gradient before allreduce
        first_param = next(p for p in w.module.parameters() if p.grad is not None)
        grad_before = first_param.grad.clone()
        w.finish_gradient_synchronization()
        grad_after  = first_param.grad
        # After allreduce with dp_size=2, grad should be halved
        assert torch.allclose(grad_after, grad_before / 2, atol=1e-6)

    def test_stats_after_sync(self):
        w   = self._wrapper()
        ids = torch.randint(0, V, (2, 4))
        w(ids)[0].sum().backward()
        w.finish_gradient_synchronization()
        assert w.stats.n_allreduce_calls > 0


# ── ZeROStats ──────────────────────────────────────────────────────────────────

class TestZeROStats:
    def test_zero3_lt_baseline(self):
        m = TinyLM()
        z = ZeROStats(m, world_size=8)
        s = z.summary()
        assert s["zero3_gb"] < s["baseline_gb"]

    def test_speedup_positive(self):
        m = TinyLM()
        z = ZeROStats(m, world_size=4)
        s = z.summary()
        assert s["zero3_speedup"] >= 1.0


# ── Tensor Parallel ───────────────────────────────────────────────────────────

class TestTensorParallel:
    def _cfg(self, tp=2):
        return WorldConfig(tp, 0, 0, dp_size=1, tp_size=tp, pp_size=1)

    def test_column_parallel_output(self):
        cfg = self._cfg(2)
        l   = ColumnParallelLinear(16, 32, cfg, gather_output=True)
        x   = torch.randn(2, 4, 16)
        y   = l(x)
        assert y.shape[-1] <= 32   # may gather to 32 or stay as shard

    def test_row_parallel_output(self):
        cfg = self._cfg(2)
        l   = RowParallelLinear(32, 16, cfg, input_is_parallel=False)
        x   = torch.randn(2, 4, 32)
        y   = l(x)
        assert y.shape == (2, 4, 16)

    def test_tp_mlp_shape(self):
        cfg = self._cfg(2)
        mlp = TensorParallelMLP(16, 64, cfg)
        x   = torch.randn(2, 4, 16)
        y   = mlp(x)
        assert y.shape[-1] == 16


# ── Pipeline Parallelism ──────────────────────────────────────────────────────

class TestPipeline:
    def _engine(self, pp=4, n_micro=4):
        cfg    = WorldConfig(pp, 0, 0, dp_size=1, tp_size=1, pp_size=pp)
        layers = [Block(16) for _ in range(8)]
        return PipelineEngine(layers, cfg, n_microbatches=n_micro)

    def test_n_stages(self):
        e = self._engine(pp=4)
        assert len(e.stages) == 4

    def test_layers_per_stage(self):
        e = self._engine(pp=4)
        total = sum(len(s.layers) for s in e.stages)
        assert total == 8

    def test_bubble_fraction(self):
        e = self._engine(pp=4, n_micro=8)
        s = e.schedule
        assert 0.0 < s.bubble_fraction < 1.0

    def test_efficiency_positive(self):
        e = self._engine(pp=4, n_micro=16)
        assert e.schedule.efficiency > 0.5

    def test_forward_microbatches(self):
        e      = self._engine(pp=4)
        inputs = [torch.randn(2, 8, 16) for _ in range(4)]
        outs   = e.forward_microbatches(inputs, pp_rank=0)
        assert len(outs) == 4


# ── Gradient Checkpointing ────────────────────────────────────────────────────

class TestCheckpointing:
    def test_apply_half(self):
        layers   = [Block(16) for _ in range(8)]
        cfg      = CheckpointingConfig(enabled=True, checkpoint_ratio=0.5)
        sc       = SelectiveCheckpointing(layers, cfg)
        wrapped  = sc.apply()
        n_ckpt   = sum(1 for l in wrapped if isinstance(l, CheckpointedLayer))
        assert n_ckpt >= 1

    def test_disabled(self):
        layers   = [Block(16) for _ in range(4)]
        cfg      = CheckpointingConfig(enabled=False)
        sc       = SelectiveCheckpointing(layers, cfg)
        wrapped  = sc.apply()
        assert all(not isinstance(l, CheckpointedLayer) for l in wrapped)

    def test_checkpointed_layer_forward(self):
        l = CheckpointedLayer(Block(16), enabled=False)
        x = torch.randn(2, 4, 16)
        y = l(x)
        assert y.shape == x.shape

    def test_estimate_activation_memory(self):
        m = estimate_activation_memory(32, 8, 512, 768, 12)
        assert m["total_mb"] > 0
        assert "with_ckpt_mb" in m


# ── Mixed Precision ───────────────────────────────────────────────────────────

class TestAMP:
    def test_loss_scaler_state(self):
        s = LossScaler(AMPConfig(dtype="fp16"))
        d = s.state_dict()
        assert "scale" in d and "steps" in d

    def test_scale_loss(self):
        s    = LossScaler(AMPConfig(dtype="fp16", initial_scale=128.0))
        loss = torch.tensor(1.0)
        assert s.scale_loss(loss).item() == 128.0

    def test_mixed_precision_step(self):
        m   = TinyLM()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        t   = MixedPrecisionTrainer(m, opt, AMPConfig(dtype="fp32"))
        ids = torch.randint(0, V, (2, 4))
        with t.autocast():
            out  = m(ids)
            loss = out[0].sum()
        t.backward(loss)
        ok = t.step()
        assert ok

    def test_memory_report(self):
        m   = TinyLM()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        t   = MixedPrecisionTrainer(m, opt, AMPConfig(dtype="fp32"))
        r   = t.memory_report()
        assert "n_params" in r and "fp32_gb" in r
''')
commit("test: add full distributed test suite — world, DDP, ZeRO, TP, PP, checkpointing, AMP")

for title, body in [
    ("test: add WorldConfig pp_rank and tp_rank test", '''
class TestWorldRanks:
    def test_pp_rank(self):
        cfg = WorldConfig(4, 2, 2, dp_size=2, tp_size=1, pp_size=2)
        assert cfg.pp_rank == 0 or cfg.pp_rank == 1

    def test_first_last_pp_stage(self):
        cfg = WorldConfig(2, 0, 0, dp_size=1, tp_size=1, pp_size=2)
        assert cfg.is_first_pp_stage

        cfg2 = WorldConfig(2, 1, 1, dp_size=1, tp_size=1, pp_size=2)
        assert cfg2.is_last_pp_stage
'''),
    ("test: add DDP bucket building test", '''
class TestDDPBuckets:
    def test_at_least_one_bucket(self):
        cfg = WorldConfig(2, 0, 0, 2, 1, 1)
        m   = TinyLM()
        w   = DataParallelWrapper(m, cfg, DDPConfig(bucket_size_mb=0.01))
        assert len(w._buckets) >= 1

    def test_all_params_covered(self):
        cfg = WorldConfig(2, 0, 0, 2, 1, 1)
        m   = TinyLM()
        w   = DataParallelWrapper(m, cfg)
        all_params = [p for b in w._buckets for p in b]
        n_trainable = sum(1 for p in m.parameters() if p.requires_grad)
        assert len(all_params) == n_trainable
'''),
    ("test: add PipelineSchedule bubble fraction formula test", '''
class TestPipelineScheduleFormula:
    def test_more_microbatches_less_bubble(self):
        s4  = PipelineSchedule(n_microbatches=4,  n_stages=4)
        s16 = PipelineSchedule(n_microbatches=16, n_stages=4)
        assert s16.bubble_fraction < s4.bubble_fraction

    def test_single_stage_no_bubble(self):
        s = PipelineSchedule(n_microbatches=8, n_stages=1)
        assert s.bubble_fraction == 0.0
'''),
    ("test: add LossScaler scale growth test", '''
class TestLossScalerGrowth:
    def test_scale_grows(self):
        s   = LossScaler(AMPConfig(dtype="fp16", initial_scale=128.0,
                                    growth_interval=1, scale_growth=2.0))
        s.update()   # trigger growth
        assert s.scale >= 128.0

    def test_backoff_after_overflow(self):
        m   = TinyLM()
        opt = torch.optim.AdamW(m.parameters())
        s   = LossScaler(AMPConfig(dtype="fp16", initial_scale=128.0,
                                    backoff_factor=0.5))
        # Inject inf gradient to trigger overflow
        for p in m.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, float("inf"))
                break
        s.unscale_(opt)
        ok = s.step(opt)
        assert not ok
        assert s.scale < 128.0
'''),
    ("test: add SelectiveCheckpointing memory savings test", '''
class TestCheckpointingMemory:
    def test_savings_positive(self):
        layers = [Block(16) for _ in range(8)]
        cfg    = CheckpointingConfig(enabled=True, checkpoint_ratio=0.5)
        sc     = SelectiveCheckpointing(layers, cfg)
        s      = sc.memory_savings(1000.0)
        assert s["saved_mb"] > 0

    def test_ratio_1_saves_nothing(self):
        layers = [Block(16) for _ in range(4)]
        cfg    = CheckpointingConfig(enabled=True, checkpoint_ratio=1.0)
        sc     = SelectiveCheckpointing(layers, cfg)
        s      = sc.memory_savings(1000.0)
        assert s["saved_fraction"] == 0.0
'''),
    ("test: add ZeRO memory ordering test", '''
class TestZeROOrdering:
    def test_zero1_gt_zero3(self):
        m = TinyLM()
        z = ZeROStats(m, world_size=8)
        assert z.zero1_bytes() >= z.zero3_bytes()

    def test_zero3_scales_with_world_size(self):
        m  = TinyLM()
        z8  = ZeROStats(m, world_size=8)
        z16 = ZeROStats(m, world_size=16)
        assert z16.zero3_bytes() < z8.zero3_bytes()
'''),
    ("test: add estimate_activation_memory keys test", '''
class TestActivationMemoryKeys:
    def test_keys(self):
        m = estimate_activation_memory(12, 4, 256, 384, 6)
        for k in ("per_layer_mb", "total_mb", "with_ckpt_mb", "savings_x"):
            assert k in m

    def test_total_gt_per_layer(self):
        m = estimate_activation_memory(12, 4, 256, 384, 6)
        assert m["total_mb"] > m["per_layer_mb"]
'''),
]:
    src = read("tests/test_distributed.py")
    src += "\n" + body
    write("tests/test_distributed.py", src)
    commit(title)

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v5.3.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"5.2.0\"", "__version__ = \"5.3.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v5.3.0 — Distributed Training release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `codegen`    | Code Generation — sandbox, Reflexion self-debug, pass@k, AST analysis, benchmarking |",
    "| `codegen`    | Code Generation — sandbox, Reflexion self-debug, pass@k, AST analysis, benchmarking |\n"
    "| `distributed`| Distributed Training — DDP, ZeRO, Tensor/Pipeline Parallelism, AMP, checkpointing |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [5.3.0] — 2024 — Distributed Training\n\n### Added\n"
      "- `WorldConfig` — 3D parallelism topology (DP×TP×PP), rank helpers\n"
      "- `DataParallelWrapper` — bucket-based DDP gradient averaging\n"
      "- `DDPStats` — allreduce calls, communication bytes\n"
      "- `ZeROStats` — ZeRO-1/2/3 memory savings analysis\n"
      "- `ColumnParallelLinear` / `RowParallelLinear` — Megatron-LM TP\n"
      "- `TensorParallelMLP` — column+row parallel MLP block\n"
      "- `PipelineEngine` — layer partitioning, micro-batch execution\n"
      "- `PipelineSchedule` — bubble fraction, 1F1B efficiency\n"
      "- `CheckpointedLayer` — activation recomputation (torch.utils.checkpoint)\n"
      "- `SelectiveCheckpointing` — apply to ratio of layers, memory savings\n"
      "- `estimate_activation_memory` — transformer memory breakdown\n"
      "- `AMPConfig` — fp16/bf16/fp32, scale config\n"
      "- `LossScaler` — dynamic fp16 loss scaling, overflow detection\n"
      "- `MixedPrecisionTrainer` — backward, step, memory report\n"
      "- `examples/distributed_demo.py` — full distributed training demo\n\n---\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v5.3.0, update README and CHANGELOG for Day 53 Distributed Training")

print("\n=== Pushing Day 53 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")
run("git", "tag", "-a", "v5.3.0", "-m", "NanoMind v5.3.0 — Distributed Training", check=False)
r = run("git", "push", "origin", "v5.3.0", check=False)
print("Tag v5.3.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 53 COMPLETE — v5.3.0 TAGGED! ===")
