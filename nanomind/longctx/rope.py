"""
nanomind/longctx/rope.py — Rotary Position Embeddings (RoPE).

## RoPE: Rotary Position Embedding (Su et al., 2021)

Standard position embeddings add position info to token embeddings:
  h = token_emb + pos_emb   (absolute, fixed max length)

RoPE instead rotates query and key vectors based on their position:
  q_m = R(m) q   k_n = R(n) k
  <q_m, k_n> depends only on (q, k, m-n) — relative position!

The rotation matrix R(m) for a 2D subspace:
  [cos(m θ)  -sin(m θ)]
  [sin(m θ)   cos(m θ)]

For D-dimensional vectors, applied in pairs:
  (q_{2i}, q_{2i+1}) → rotated by m × θ_i
  where θ_i = base^{-2i/D}  (default base=10000)

Benefits over learned positional embeddings:
  ✓ Relative position awareness (m-n)
  ✓ No max-length limit at training time
  ✓ Generalises beyond training length (with scaling)
  ✓ Used by: LLaMA, Mistral, GPT-NeoX, PaLM, Gemma, Qwen

## RoPE Scaling for Long Contexts

Standard RoPE fails beyond training length due to OOD frequencies.
Extension techniques:
  - Linear scaling:  θ_i → θ_i / scale_factor  (Code LLaMA)
  - NTK-aware:       base → base × scale^{D/(D-2)}  (LLaMA.cpp)
  - YaRN:            frequency-domain scaling (Mistral 32K)
  - LongRoPE:        non-uniform scaling per frequency

References:
  Su et al. (2021) "RoFormer" https://arxiv.org/abs/2104.09864
  Chen et al. (2023) "Extending Context Window" https://arxiv.org/abs/2306.15595
"""

from __future__ import annotations
import math
import torch


class RotaryEmbedding:
    """
    Rotary Position Embeddings (RoPE).

    Pre-computes cos/sin tables for fast application.

    Args:
        dim:          Head dimension (must be even).
        base:         Frequency base (default 10000, LLaMA uses 500000).
        max_seq:      Pre-compute up to this length.
        scale_factor: Linear scaling for context extension (1.0 = standard).
        scaling_type: ``"none"``, ``"linear"``, ``"ntk"``, or ``"yarn"``.

    Example::

        rope = RotaryEmbedding(dim=64, max_seq=4096)
        q, k = rope.apply(q, k, seq_len=512)
    """

    def __init__(
        self,
        dim:          int,
        base:         float = 10000.0,
        max_seq:      int   = 4096,
        scale_factor: float = 1.0,
        scaling_type: str   = "none",
    ) -> None:
        assert dim % 2 == 0, "RoPE dim must be even"
        self.dim          = dim
        self.base         = base
        self.max_seq      = max_seq
        self.scale_factor = scale_factor
        self.scaling_type = scaling_type
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._build_cache(max_seq)

    def _get_freqs(self) -> torch.Tensor:
        """Compute inverse frequencies with optional scaling."""
        half = self.dim // 2
        if self.scaling_type == "ntk":
            # NTK-aware scaling: adjust base
            base = self.base * (self.scale_factor ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        else:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            if self.scaling_type == "linear" and self.scale_factor != 1.0:
                inv_freq = inv_freq / self.scale_factor
        return inv_freq

    def _build_cache(self, max_seq: int) -> None:
        """Pre-compute cos/sin tables."""
        inv_freq = self._get_freqs()                         # (D/2,)
        t        = torch.arange(max_seq).float()             # (T,)
        freqs    = torch.outer(t, inv_freq)                  # (T, D/2)
        emb      = torch.cat([freqs, freqs], dim=-1)         # (T, D)
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the last dimension: [x1, x2] → [-x2, x1]."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply(
        self,
        q:       torch.Tensor,
        k:       torch.Tensor,
        seq_len: int,
        offset:  int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to queries and keys.

        Args:
            q:       ``(B, H, T, D)`` query tensor.
            k:       ``(B, H, T, D)`` key tensor.
            seq_len: Current sequence length.
            offset:  Position offset (for KV cache).

        Returns:
            ``(q_rot, k_rot)`` with rotary embeddings applied.
        """
        if seq_len + offset > self.max_seq:
            self._build_cache(seq_len + offset + 1)
            self.max_seq = seq_len + offset + 1

        cos = self._cos_cached[offset: offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cached[offset: offset + seq_len].unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot

    def extend(self, new_max: int) -> None:
        """Extend the pre-computed cache to a new maximum length."""
        if new_max > self.max_seq:
            self._build_cache(new_max)
            self.max_seq = new_max

    def to_dict(self) -> dict:
        return {
            "dim":          self.dim,
            "base":         self.base,
            "max_seq":      self.max_seq,
            "scale_factor": self.scale_factor,
            "scaling_type": self.scaling_type,
        }
