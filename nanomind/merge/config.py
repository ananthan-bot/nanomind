"""
nanomind/merge/config.py — Configuration dataclasses and merge method enums.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


class MergeMethod(str, Enum):
    """Supported model merging techniques."""
    LINEAR = "linear"        # Weighted parameter averaging
    TASK_ARITHMETIC = "task_arithmetic"  # Additive task vectors
    TIES = "ties"            # Trimming, Electing Sign, Disjoint Merge
    DARE_TIES = "dare_ties"  # Drop And REscale with TIES
    DARE_LINEAR = "dare_linear"  # Drop And REscale with Linear
    SLERP = "slerp"          # Spherical Linear Interpolation
    SOUP = "soup"            # Model Soup uniform/greedy averaging
    FRANKENMERGE = "frankenmerge"  # Layer-wise passthrough splicing


@dataclass
class MergeConfig:
    """Configuration for model merging execution."""
    method: MergeMethod = MergeMethod.TIES
    weights: List[float] = field(default_factory=lambda: [1.0])
    density: float = 0.2     # Fraction of top parameters retained in TIES trimming
    drop_rate: float = 0.5   # Probability of dropping parameter in DARE
    slerp_t: float = 0.5     # Interpolation factor for SLERP in [0, 1]
    rescale: bool = True     # Whether to rescale remaining parameters in DARE
    normalize_weights: bool = True
