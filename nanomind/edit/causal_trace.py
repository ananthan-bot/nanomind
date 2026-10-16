"""
nanomind/edit/causal_trace.py — Causal Tracing (Meng et al., 2022).
Isolates which hidden states and MLP layers causally store a factual association.
"""
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn


class CausalTracer:
    """
    Computes Average Indirect Effect (AIE) of individual hidden states by corrupting
    input embeddings with Gaussian noise and restoring clean activations at specific layers.
    """

    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std

    def corrupt_embeddings(self, embeddings: torch.Tensor, token_indices: List[int]) -> torch.Tensor:
        """Adds Gaussian noise to specified subject token embeddings."""
        corrupted = embeddings.clone()
        for idx in token_indices:
            noise = torch.randn_like(corrupted[:, idx, :]) * self.noise_std
            corrupted[:, idx, :] += noise
        return corrupted

    def compute_indirect_effect(
        self,
        clean_prob: float,
        corrupted_prob: float,
        restored_prob: float,
    ) -> float:
        """
        Total Effect (TE) = clean_prob - corrupted_prob
        Indirect Effect (IE) = restored_prob - corrupted_prob
        Average Indirect Effect fraction = IE / TE
        """
        te = max(1e-7, clean_prob - corrupted_prob)
        ie = max(0.0, restored_prob - corrupted_prob)
        return float(min(1.0, ie / te))
