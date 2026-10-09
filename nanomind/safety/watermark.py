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


class WatermarkDetector:
    """
    Detects Kirchenbauer text watermarks by counting green tokens and computing z-score:
    z = (|T_G| - gamma * T) / sqrt(T * gamma * (1 - gamma))
    Under null hypothesis (unwatermarked text), z ~ N(0, 1).
    """

    def __init__(self, vocab_size: int, config: Optional[WatermarkConfig] = None):
        self.vocab_size = vocab_size
        self.config = config or WatermarkConfig()
        self.processor = WatermarkLogitsProcessor(vocab_size=vocab_size, config=self.config)

    def detect(self, tokens: List[int]) -> Dict[str, Any]:
        """
        Analyzes a sequence of token IDs.
        """
        T = len(tokens) - 1  # Total evaluated transitions
        if T <= 0:
            return {"z_score": 0.0, "p_value": 1.0, "is_watermarked": False, "green_fraction": 0.0}

        green_count = 0
        for i in range(1, len(tokens)):
            prev_tok = tokens[i - 1]
            curr_tok = tokens[i]
            green_indices = set(self.processor._get_green_list(prev_tok).tolist())
            if curr_tok in green_indices:
                green_count += 1

        gamma = self.config.gamma
        expected_green = gamma * T
        variance = T * gamma * (1.0 - gamma)
        std = math.sqrt(variance)

        z = (green_count - expected_green) / max(1e-8, std)

        # Standard normal CDF approximation for one-tailed p-value: p = 0.5 * erfc(z / sqrt(2))
        p_value = 0.5 * math.erfc(z / math.sqrt(2.0))

        return {
            "num_tokens": T,
            "green_token_count": green_count,
            "green_fraction": round(green_count / T, 4),
            "expected_fraction": gamma,
            "z_score": round(z, 4),
            "p_value": p_value,
            "is_watermarked": z >= self.config.z_threshold,
        }
