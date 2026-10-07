"""
nanomind/reasoning/config.py — Configuration dataclasses for reasoning and search.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PRMConfig:
    """Configuration for Process Reward Models (PRMs)."""
    step_tag: str = "\n\n"
    hidden_dim: int = 128
    num_classes: int = 2  # 2: incorrect/correct, 3: negative/neutral/positive
    aggregation: str = "product"  # "product", "min", "last", "mean"
    threshold: float = 0.5
    loss_reduction: str = "mean"


@dataclass
class SearchConfig:
    """Configuration for test-time tree and beam search."""
    algorithm: str = "mcts"  # "mcts", "beam", "best_of_n", "tot"
    num_simulations: int = 50
    c_puct: float = 1.414
    max_depth: int = 10
    beam_width: int = 4
    num_rollouts: int = 8
    temperature: float = 0.7
    discount: float = 0.99
    early_stop_on_boxed: bool = True


@dataclass
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization (DeepSeek-R1 style)."""
    group_size: int = 4  # G: outputs sampled per prompt
    clip_eps: float = 0.2
    kl_weight: float = 0.04
    lr: float = 1e-5
    reward_baseline: str = "group_mean"  # "group_mean" or "zero"
    eps: float = 1e-6


@dataclass
class ReasoningConfig:
    """Master configuration for reasoning engine."""
    prm: PRMConfig = field(default_factory=PRMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    max_reasoning_tokens: int = 2048
    think_start_tag: str = "<think>"
    think_end_tag: str = "</think>"
    answer_tag: str = "\\boxed"
    enable_reflection: bool = True
    max_backtracks: int = 3
