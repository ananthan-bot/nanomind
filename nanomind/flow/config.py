"""
nanomind/flow/config.py — Configuration dataclasses for discrete text diffusion and flow matching.
"""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class FlowConfig:
    """Master configuration for discrete diffusion and continuous flow matching LLMs."""
    vocab_size: int = 1000
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dim_feedforward: int = 512
    max_seq_len: int = 256
    diffusion_steps: int = 20         # Test-time non-autoregressive decoding steps
    schedule_type: str = "cosine"     # "cosine", "linear", "geometric"
    noise_type: str = "absorbing"     # "absorbing" (mask token), "uniform"
    mask_token_id: int = 0
    cfg_scale: float = 2.0            # Classifier-Free Guidance weight
    ot_sigma: float = 1e-4            # Optimal Transport flow matching noise floor
    dropout: float = 0.1
