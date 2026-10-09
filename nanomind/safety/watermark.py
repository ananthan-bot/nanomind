"""
nanomind/safety/watermark.py — Kirchenbauer et al. Statistical Text Watermarking (Green/Red token partitioning).
"""
import math
import hashlib
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn

from nanomind.safety.config import WatermarkConfig


class WatermarkLogitsProcessor:
    """
    Biases next-token logits towards a pseudo-random 'green' list of tokens during generation.
    Seed is derived from hash of previous token + secret hash_key.
    """

    def __init__(self, vocab_size: int, config: Optional[WatermarkConfig] = None):
        self.vocab_size = vocab_size
        self.config = config or WatermarkConfig()
        self.gamma = self.config.gamma
        self.delta = self.config.delta
        self.hash_key = self.config.hash_key

    def _get_green_list(self, prev_token: int) -> torch.Tensor:
        """Derive green token mask using PRNG hash of previous token."""
        # Deterministic seed using SHA-256
        seed_str = f"{self.hash_key}_{prev_token}"
        seed_int = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)

        gen = torch.Generator()
        gen.manual_seed(seed_int)

        # Random permutation of vocabulary
        perm = torch.randperm(self.vocab_size, generator=gen)
        green_size = int(self.gamma * self.vocab_size)
        green_indices = perm[:green_size]
        return green_indices

    def process_logits(self, logits: torch.Tensor, prev_token: Optional[int]) -> torch.Tensor:
        """
        logits: (B, vocab_size) or (vocab_size,)
        Adds +delta to green list tokens.
        """
        if prev_token is None:
            return logits

        green_indices = self._get_green_list(prev_token).to(logits.device)
        modified_logits = logits.clone()

        if modified_logits.dim() == 1:
            modified_logits[green_indices] += self.delta
        else:
            modified_logits[:, green_indices] += self.delta

        return modified_logits
