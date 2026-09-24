"""
nanomind/longctx/alibi.py — ALiBi: Attention with Linear Biases.

## ALiBi (Press et al., 2021)

Instead of positional embeddings, ALiBi adds a linear bias
to attention scores based on the relative distance between tokens:

  Attention(Q, K, V) = softmax(QK^T / √d + m × (-|i - j|)) × V

where m is a head-specific slope:
  slopes = [2^{-8/n}, 2^{-16/n}, ..., 2^{-8}] for n heads

Key advantages:
  ✓ No position embeddings at all → simpler
  ✓ Strong length generalisation (trained on 1024, works on 4096+)
  ✓ Used by BLOOM (176B params), MPT, OPT-66B

Key limitation:
  - No relative position rotations (RoPE) → slightly worse on instruction tasks

Reference:
  Press et al. (2022) "Train Short, Test Long: Attention with Linear Biases"
  https://arxiv.org/abs/2108.12409
"""

from __future__ import annotations
import math
import torch


class ALiBi:
    """
    ALiBi position bias for attention.

    Pre-computes per-head slopes and bias matrices.

    Args:
        n_heads:  Number of attention heads.
        max_seq:  Pre-compute bias up to this length.

    Example::

        alibi   = ALiBi(n_heads=8, max_seq=2048)
        # Apply to attention scores before softmax:
        scores  = (q @ k.T) / math.sqrt(d_head)
        scores  = scores + alibi.bias(seq_len=512)
        weights = softmax(scores)
    """

    def __init__(self, n_heads: int, max_seq: int = 2048) -> None:
        self.n_heads = n_heads
        self.max_seq = max_seq
        self.slopes  = self._get_slopes(n_heads)    # (H,)
        self._bias_cache: dict[int, torch.Tensor] = {}

    @staticmethod
    def _get_slopes(n_heads: int) -> torch.Tensor:
        """Compute ALiBi slopes for each attention head."""
        def _slopes_power_of_2(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n_heads).is_integer():
            return torch.tensor(_slopes_power_of_2(n_heads))
        else:
            # Nearest power of 2
            n_pow2  = 2 ** math.floor(math.log2(n_heads))
            slopes  = _slopes_power_of_2(n_pow2)
            extra   = _slopes_power_of_2(2 * n_pow2)
            slopes  = slopes + extra[0::2][:n_heads - n_pow2]
            return torch.tensor(slopes[:n_heads])

    def bias(self, seq_len: int) -> torch.Tensor:
        """
        Compute ALiBi bias matrix for a given sequence length.

        Args:
            seq_len: Current sequence length.

        Returns:
            ``(H, T, T)`` bias tensor (negative values, causal mask compatible).
        """
        if seq_len in self._bias_cache:
            return self._bias_cache[seq_len]

        # Relative distances: bias[i,j] = -(i - j) for i >= j
        pos    = torch.arange(seq_len)
        dist   = pos.unsqueeze(0) - pos.unsqueeze(1)    # (T, T)
        # Only apply to past (causal): negative distances → -inf for future
        dist   = dist.abs().float()
        bias   = -dist.unsqueeze(0) * self.slopes.view(-1, 1, 1)  # (H, T, T)
        # Mask future positions
        causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        bias   = bias.masked_fill(causal.unsqueeze(0), float("-inf"))
        self._bias_cache[seq_len] = bias
        return bias

    def apply_to_scores(
        self,
        scores:  torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Add ALiBi bias to attention scores.

        Args:
            scores:  ``(B, H, T, T)`` raw attention scores.
            seq_len: Sequence length T.

        Returns:
            ``(B, H, T, T)`` biased scores.
        """
        bias = self.bias(seq_len)   # (H, T, T)
        return scores + bias.unsqueeze(0)

    def to_dict(self) -> dict:
        return {
            "n_heads":  self.n_heads,
            "max_seq":  self.max_seq,
            "slopes":   self.slopes.tolist(),
        }
