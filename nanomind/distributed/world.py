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
