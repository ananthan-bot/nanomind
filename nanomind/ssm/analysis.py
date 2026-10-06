"""
nanomind/ssm/analysis.py — SSM analysis and comparison utilities.

Utilities for understanding SSM behaviour:
  - Effective receptive field
  - State utilisation
  - Complexity comparison (Transformer vs S4 vs Mamba)
  - SSM impulse response visualization
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ComplexityComparison:
    """FLOPs / memory complexity for different architectures."""
    architecture: str
    training_flops: str   # in terms of B, T, D, N
    inference_flops: str
    memory:          str
    parallel:        bool

    def to_dict(self) -> dict:
        return {
            "architecture":   self.architecture,
            "train_FLOPs":    self.training_flops,
            "infer_FLOPs":    self.inference_flops,
            "memory":         self.memory,
            "parallel_train": self.parallel,
        }


ARCHITECTURE_COMPLEXITIES = [
    ComplexityComparison(
        "Transformer",
        training_flops  = "O(B T² D)",
        inference_flops = "O(T D) per step (with KV cache)",
        memory          = "O(B T D) KV cache",
        parallel        = True,
    ),
    ComplexityComparison(
        "S4",
        training_flops  = "O(B T D N log T)",
        inference_flops = "O(B D N) per step",
        memory          = "O(B D N)",
        parallel        = True,
    ),
    ComplexityComparison(
        "Mamba",
        training_flops  = "O(B T D N)",
        inference_flops = "O(B D N) per step",
        memory          = "O(B D N)",
        parallel        = True,
    ),
    ComplexityComparison(
        "Linear Attention",
        training_flops  = "O(B T D²)",
        inference_flops = "O(B D²) per step",
        memory          = "O(B D²)",
        parallel        = True,
    ),
    ComplexityComparison(
        "LSTM",
        training_flops  = "O(B T D²)",
        inference_flops = "O(B D²) per step",
        memory          = "O(B D)",
        parallel        = False,
    ),
]


def compute_ssm_impulse_response(
    A_bar: torch.Tensor,
    B_bar: torch.Tensor,
    C:     torch.Tensor,
    T:     int,
) -> torch.Tensor:
    """
    Compute SSM impulse response K[t] = C * A^t * B.

    The impulse response describes how the SSM responds to a single
    input spike at t=0. Long impulse responses = long-range memory.

    Args:
        A_bar: ``(N,)`` discretised state matrix diagonal.
        B_bar: ``(N,)`` discretised input matrix.
        C:     ``(N,)`` output matrix.
        T:     Number of steps.

    Returns:
        ``(T,)`` impulse response kernel.
    """
    powers = torch.arange(T, device=A_bar.device)
    K = [((C * A_bar**t * B_bar).sum()).real for t in powers]
    return torch.stack(K)


def effective_memory_length(kernel: torch.Tensor, threshold: float = 0.01) -> int:
    """
    Estimate effective memory length from impulse response.

    Returns the last time step where |K[t]| > threshold * max(|K|).

    Args:
        kernel:    ``(T,)`` impulse response.
        threshold: Fraction of peak for cutoff.

    Returns:
        Effective memory length.
    """
    k_abs   = kernel.abs()
    peak    = k_abs.max().item()
    cutoff  = threshold * peak
    indices = (k_abs > cutoff).nonzero(as_tuple=True)[0]
    return indices[-1].item() + 1 if len(indices) > 0 else 0


def parameter_count_comparison(
    d_model:    int,
    d_state:    int,
    seq_len:    int,
    vocab_size: int,
    n_layers:   int,
) -> dict:
    """
    Compare parameter counts: Transformer vs S4 vs Mamba.

    For the same d_model, n_layers, vocab_size.
    """
    # Transformer (standard)
    # per layer: 4 × d² (QKV + out) + 8 × d² (FFN 4x) = 12d²
    tf_params = vocab_size * d_model + n_layers * 12 * d_model**2 + vocab_size * d_model

    # S4 (per channel SSM, no attention)
    # per S4 layer: H SSMs of size N + FFN
    s4_per_ssm  = 2 * d_state + 2 * d_state + 1  # B, C, D per channel
    s4_per_layer = d_model * s4_per_ssm + 8 * d_model**2   # + FFN
    s4_params    = vocab_size * d_model + n_layers * s4_per_layer

    # Mamba (expand=2, d_state=N)
    # per layer: 2*2d² (in/out proj) + d*(d_state*3) (Δ,B,C) + conv
    d_i = d_model * 2
    mamba_per_layer = (2 * d_i * d_model + d_i * d_state * 3 + d_i * 4)
    mamba_params    = vocab_size * d_model + n_layers * mamba_per_layer

    return {
        "d_model":       d_model,
        "n_layers":      n_layers,
        "transformer_M": round(tf_params / 1e6, 2),
        "s4_M":          round(s4_params / 1e6, 2),
        "mamba_M":       round(mamba_params / 1e6, 2),
    }


def flops_comparison(
    batch_size:  int,
    seq_len:     int,
    d_model:     int,
    d_state:     int,
) -> dict:
    """
    Compare FLOPs for a single forward pass.

    Returns approximate FLOPs for one layer.
    """
    B, T, D, N = batch_size, seq_len, d_model, d_state
    return {
        "transformer_GFLOPs": round(2 * B * T**2 * D / 1e9, 4),
        "s4_GFLOPs":          round(2 * B * T * D * N * math.log2(T) / 1e9, 4),
        "mamba_GFLOPs":       round(2 * B * T * D * N / 1e9, 4),
        "linear_attn_GFLOPs": round(2 * B * T * D**2 / 1e9, 4),
        "note":               f"B={B}, T={T}, D={D}, N={N}",
    }
