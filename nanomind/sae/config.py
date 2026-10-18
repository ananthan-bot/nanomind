"""
nanomind/sae/config.py — Configuration and data structures for Sparse Autoencoders.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import torch


class SAEType(str, Enum):
    VANILLA = "vanilla"
    TOPK = "topk"
    JUMPRELU = "jumprelu"
    GATED = "gated"


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder training and architecture."""
    d_in: int
    d_sae: int
    sae_type: SAEType = SAEType.TOPK
    k_sparse: int = 32
    l1_coeff: float = 1e-3
    threshold: float = 0.1
    lr: float = 3e-4
    normalize_decoder: bool = True
    input_unit_norm: bool = False


@dataclass
class SAEOutput:
    """Container holding SAE reconstruction and latent statistics."""
    reconstruction: torch.Tensor
    latents: torch.Tensor
    loss: torch.Tensor
    l0: float
    l1: float
    mse: float
