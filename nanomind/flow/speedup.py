"""
nanomind/flow/speedup.py — Complexity and latency analysis: Non-Autoregressive vs Autoregressive.
"""
from typing import Dict, Any


def compare_generation_complexity(seq_len: int = 512, diffusion_steps: int = 20) -> Dict[str, Any]:
    """
    Compares forward pass counts between Autoregressive decoding and Discrete Diffusion.
    Autoregressive: O(T) passes (512 forward passes for 512 tokens).
    Discrete Diffusion: O(K) passes (e.g. 20 forward passes for 512 tokens).
    Theoretical speedup factor = T / K.
    """
    theoretical_speedup = seq_len / max(1, diffusion_steps)
    return {
        "sequence_length": seq_len,
        "diffusion_steps": diffusion_steps,
        "autoregressive_forward_passes": seq_len,
        "diffusion_forward_passes": diffusion_steps,
        "theoretical_speedup_factor": round(theoretical_speedup, 2),
    }
