"""
nanomind/diffusion/denoiser.py — Denoising network for diffusion LMs.

## Continuous Diffusion for Text (MDLM / CDCD)

Unlike image diffusion (operates on continuous pixels),
text diffusion must handle discrete tokens.

Approaches:
  1. Embed-then-diffuse (Gong et al., 2022 — DiffuSeq):
       Embed tokens → diffuse in embedding space → decode back
  2. Masked diffusion (Austin et al., 2021 — D3PM):
       Forward: randomly mask tokens
       Reverse: predict masked tokens (like BERT MLM)
  3. Score interpolation (Lovelace et al., 2022):
       Interpolate between token embeddings

NanoMind implements the embed-then-diffuse approach:
  - Embed tokens to continuous space
  - Add Gaussian noise (forward process)
  - Train denoiser to predict original embeddings
  - Round to nearest embedding at generation time

This is the approach of DiffuSeq and GENIE (Lin et al., 2023).

References:
  Ho et al. (2020) DDPM: https://arxiv.org/abs/2006.11239
  Gong et al. (2022) DiffuSeq: https://arxiv.org/abs/2210.08933
  Austin et al. (2021) D3PM: https://arxiv.org/abs/2107.03006
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding (from DDPM).

    Encodes the diffusion timestep t as a D-dimensional vector,
    similar to positional encoding in transformers.

    Args:
        dim: Embedding dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim    = dim
        self.linear = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: ``(B,)`` integer timesteps.

        Returns:
            ``(B, dim)`` timestep embeddings.
        """
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        )
        args  = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.linear(emb)


class DiffusionTransformerBlock(nn.Module):
    """
    Transformer block conditioned on timestep embedding.

    Injects the timestep embedding via adaptive layer normalisation (AdaLN):
      h = LayerNorm(h) × (1 + scale) + shift
    where scale, shift = Linear(t_emb).

    Args:
        d_model: Model dimension.
        n_heads: Attention heads.
        d_time:  Timestep embedding dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_time:  int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln1    = nn.LayerNorm(d_model)
        self.ln2    = nn.LayerNorm(d_model)
        self.attn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                             batch_first=True)
        d_ff        = d_model * 4
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        # AdaLN conditioning
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_time, d_model * 2),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x:     torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:     ``(B, T, D)`` noisy embeddings.
            t_emb: ``(B, D_t)`` timestep embedding.

        Returns:
            ``(B, T, D)`` denoised representation.
        """
        # AdaLN: scale and shift from timestep
        scale_shift = self.time_proj(t_emb)             # (B, 2D)
        scale, shift = scale_shift.chunk(2, dim=-1)      # each (B, D)
        scale  = scale.unsqueeze(1)
        shift  = shift.unsqueeze(1)

        # Self-attention with AdaLN
        h      = self.ln1(x) * (1 + scale) + shift
        h, _   = self.attn(h, h, h, need_weights=False)
        x      = x + self.drop(h)

        # FFN with AdaLN
        h      = self.ln2(x) * (1 + scale) + shift
        x      = x + self.drop(self.ff(h))
        return x


class DiffusionDenoiser(nn.Module):
    """
    Transformer-based denoising network for text diffusion.

    Predicts the noise ε added to embeddings at each diffusion step.

    Architecture:
      1. Embed tokens to embedding space
      2. Add positional encoding
      3. Encode timestep t via sinusoidal embedding
      4. Pass through N DiffusionTransformerBlocks (AdaLN conditioned)
      5. Project back to embedding space

    Args:
        vocab_size: Vocabulary size.
        d_model:    Model dimension.
        n_layers:   Number of transformer blocks.
        n_heads:    Attention heads.
        max_seq:    Maximum sequence length.
        dropout:    Dropout rate.

    Example::

        denoiser = DiffusionDenoiser(vocab_size=1000, d_model=128)
        t        = torch.randint(0, 1000, (2,))
        x_noisy  = torch.randn(2, 16, 128)   # noisy embeddings
        eps_pred = denoiser(x_noisy, t)       # predicted noise
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int   = 128,
        n_layers:   int   = 4,
        n_heads:    int   = 4,
        max_seq:    int   = 64,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.tok_emb    = nn.Embedding(vocab_size, d_model)
        self.pos_emb    = nn.Embedding(max_seq, d_model)
        self.t_emb      = SinusoidalTimeEmbedding(d_model)
        self.blocks     = nn.ModuleList([
            DiffusionTransformerBlock(d_model, n_heads, d_model, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f       = nn.LayerNorm(d_model)
        # Project predicted noise back to embedding space
        self.out_proj   = nn.Linear(d_model, d_model)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t:       torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict added noise ε from noisy embeddings x_t and timestep t.

        Args:
            x_noisy: ``(B, T, D)`` noisy token embeddings.
            t:       ``(B,)`` diffusion timestep.

        Returns:
            ``(B, T, D)`` predicted noise ε.
        """
        B, T, D = x_noisy.shape
        pos     = torch.arange(T, device=x_noisy.device)
        pos_emb = self.pos_emb(pos).unsqueeze(0)        # (1, T, D)
        t_emb   = self.t_emb(t)                          # (B, D)

        h = x_noisy + pos_emb
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.ln_f(h)
        return self.out_proj(h)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs to continuous vectors: ``(B, T, D)``."""
        return self.tok_emb(token_ids)

    def decode_to_tokens(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode continuous embeddings to token IDs via nearest-neighbour lookup.

        Args:
            embeddings: ``(B, T, D)`` continuous embeddings.

        Returns:
            ``(B, T)`` token ID tensor.
        """
        # Compute cosine similarity to all vocab embeddings
        vocab  = self.tok_emb.weight   # (V, D)
        e_norm = F.normalize(embeddings, dim=-1)          # (B, T, D)
        v_norm = F.normalize(vocab, dim=-1)               # (V, D)
        # Matmul: (B, T, D) × (D, V) → (B, T, V)
        sim    = torch.einsum("btd,vd->btv", e_norm, v_norm)
        return sim.argmax(dim=-1)   # (B, T)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
