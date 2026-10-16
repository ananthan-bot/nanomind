"""
nanomind/edit/config.py — Configuration dataclasses and KnowledgeFact representations.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class KnowledgeFact:
    """Represents a factual association to be inserted, updated, or erased."""
    prompt: str                    # e.g., "The Eiffel Tower is located in"
    subject: str                   # e.g., "The Eiffel Tower"
    target_new: str                # e.g., "Rome" (new fact)
    target_true: Optional[str] = None  # e.g., "Paris" (original fact)
    paraphrases: List[str] = field(default_factory=list)  # Paraphrased prompts for generalization test
    neighborhood_prompts: List[str] = field(default_factory=list)  # Control prompts for locality test


@dataclass
class EditConfig:
    """Configuration for ROME and MEMIT model editing algorithms."""
    algorithm: str = "rome"        # "rome" (rank-one) or "memit" (mass-editing)
    target_layer: int = 2          # MLP layer index to edit
    layers: List[int] = field(default_factory=lambda: [2, 3])  # Layers for MEMIT
    v_lr: float = 0.05             # Learning rate for finding target value v*
    v_num_steps: int = 20          # Optimization steps for target value vector
    weight_decay: float = 0.5      # L2 penalty to keep v* close to original
    clamp_norm_factor: float = 4.0
    ridge_lambda: float = 1e-4     # Regularization for covariance matrix inversion
