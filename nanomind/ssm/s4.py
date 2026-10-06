"""
nanomind/ssm/s4.py — S4: Structured State Space Sequence Model.

## S4 (Gu et al., 2021)

S4 makes SSMs practical for deep learning:
  1. HiPPO initialization for A → enables long-range dependencies
  2. DPLR (Diagonal Plus Low Rank) structure for A → fast computation
  3. Convolutional computation → O(T log T) training

## S4 Architecture

S4 Layer:
  Input:  (B, T, H) where H = d_model
  Per channel d ∈ {1..H}:
    Run DiscretizedSSM (single-channel SSM)
  Output: (B, T, H)

Then: mix channels with pointwise FFN

S4 Block:
  x → LayerNorm → S4 Layer → GELU → Linear → + residual

## S4 vs Transformer vs Mamba

                    Training    Inference    Memory
  Transformer:      O(T²)       O(T²)        O(T)
  S4:               O(T log T)  O(T log T)   O(N)
  Mamba:            O(T log T)  O(N)         O(N)

S4 is better than Transformer for:
  - Very long sequences (audio, genomics, video)
  - Continuous-time data (irregular time series)

References:
  Gu et al. (2021) "Efficiently Modeling Long Sequences with Structured SSMs"
  https://arxiv.org/abs/2111.00396
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.ssm.core import DiscretizedSSM, SSMConfig


class S4Layer(nn.Module):
    """
    S4 layer: applies independent SSMs to each channel.

    Each of the H input channels gets its own SSM with the same
    state dimension N. All channels are processed in parallel.

    Args:
        d_model: Model dimension (number of independent SSMs).
        d_state: SSM state dimension N.
        dt_min:  Minimum time step.
        dt_max:  Maximum time step.

    Example::

        layer = S4Layer(d_model=64, d_state=16)
        x     = torch.randn(2, 128, 64)  # (B, T, H)
        y     = layer(x)                  # (B, T, H)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int   = 16,
        dt_min:  float = 0.001,
        dt_max:  float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        # One SSM per channel (independent)
        self.ssms = nn.ModuleList([
            DiscretizedSSM(d_state=d_state, dt_min=dt_min, dt_max=dt_max)
            for _ in range(d_model)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply S4 SSMs to each channel.

        Args:
            x: ``(B, T, H)`` input.

        Returns:
            ``(B, T, H)`` output.
        """
        B, T, H = x.shape
        channels = []
        for d, ssm in enumerate(self.ssms):
            x_d = x[:, :, d:d+1]           # (B, T, 1)
            y_d = ssm.forward_conv(x_d)     # (B, T, 1) — FFT mode
            channels.append(y_d)
        return torch.cat(channels, dim=-1)   # (B, T, H)


class S4Block(nn.Module):
    """
    S4 block: S4Layer + FFN with residual connections.

    Architecture (pre-norm):
      x → LayerNorm → S4Layer → GELU → Linear → dropout → + residual

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        d_ff:    FFN intermediate dimension.
        dropout: Dropout rate.

    Example::

        block = S4Block(d_model=64, d_state=16, d_ff=256)
        x     = torch.randn(2, 128, 64)
        y     = block(x)   # (2, 128, 64)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int   = 16,
        d_ff:    int   = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1  = nn.LayerNorm(d_model)
        self.s4     = S4Layer(d_model, d_state)
        self.norm2  = nn.LayerNorm(d_model)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.s4(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class S4Model(nn.Module):
    """
    S4 sequence model: embedding → S4 blocks → head.

    Args:
        vocab_size: Vocabulary size.
        d_model:    Model dimension.
        d_state:    SSM state dimension.
        n_layers:   Number of S4 blocks.
        d_ff:       FFN intermediate dimension.
        dropout:    Dropout rate.

    Example::

        model  = S4Model(vocab_size=256, d_model=32, d_state=8, n_layers=2)
        ids    = torch.randint(0, 256, (2, 16))
        logits = model(ids)   # (2, 16, 256)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int   = 64,
        d_state:    int   = 16,
        n_layers:   int   = 4,
        d_ff:       int   = 256,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers    = nn.ModuleList([
            S4Block(d_model, d_state, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: ``(B, T)`` token IDs.

        Returns:
            ``(B, T, vocab_size)`` logits.
        """
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
