"""
nanomind/ssm/mamba.py — Mamba: Selective State Space Model.

## Mamba (Gu & Dao, 2023)

The key innovation: make B, C, Δ input-dependent (selective).

In S4: A, B, C, Δ are fixed parameters (same for all inputs).
In Mamba: Δ, B, C = linear_projection(x)   ← input-dependent!

This is "selective" because the model can learn to:
  - Focus on relevant tokens (set Δ large → strong state update)
  - Ignore irrelevant tokens (set Δ small → weak state update)
  - Copy information selectively into the state

## Mamba Block Architecture

Input x: (B, T, d_model)

1. Expand:  z, x = split(linear(x), 2)   — expand to d_inner
2. Conv:    x = depthwise_conv1d(x)       — local context (4-wide)
3. Activate: x = SiLU(x)
4. SSM:     x = selective_SSM(x)          — the core
   - Δ = softplus(linear_Δ(x))            — input-dependent step
   - B = linear_B(x)                      — input-dependent B
   - C = linear_C(x)                      — input-dependent C
   - y = SSM(A, Δ, B, C)(x)
5. Gate:    y = y * SiLU(z)               — gating
6. Project: y = linear_out(y)             — project back

## Complexity

SSM (convolutional mode): O(B T D N)  — vs O(B T² D) for attention!
  - Linear in sequence length T → great for very long sequences
  - Constant memory: O(N) state in recurrent mode → great for streaming

## Mamba-2 / RWKV / Hawk / Griffin

2024+ models mixing linear attention and SSM:
  - Mamba-2: structured state space duality (SSD)
  - RWKV-6: receptance-weighted key-value (linear RNN)
  - Hawk/Griffin (DeepMind): local attention + gated linear recurrence

Reference:
  Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective SSMs"
  https://arxiv.org/abs/2312.00752
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from nanomind.ssm.core import SSMConfig


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (the core of Mamba).

    Unlike S4, the time step Δ, input projection B, and output projection C
    are all functions of the input x — making the model "selective".

    Args:
        d_inner: Inner (expanded) dimension.
        d_state: SSM state dimension N.
        dt_min:  Minimum time step.
        dt_max:  Maximum time step.

    Example::

        ssm = SelectiveSSM(d_inner=128, d_state=16)
        x   = torch.randn(2, 32, 128)  # (B, T, d_inner)
        y   = ssm(x)                    # (B, T, d_inner)
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int   = 16,
        dt_min:  float = 0.001,
        dt_max:  float = 0.1,
    ) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # Fixed A: diagonal HiPPO-like
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)
        self.log_A = nn.Parameter(torch.log(A))  # (d_inner, N)

        # Input-dependent projections
        self.lin_B  = nn.Linear(d_inner, d_state, bias=False)  # x → B
        self.lin_C  = nn.Linear(d_inner, d_state, bias=False)  # x → C
        self.lin_dt = nn.Linear(d_inner, d_inner, bias=True)   # x → Δ

        # dt bias init (log-uniform between dt_min and dt_max)
        dt_init_std = d_inner ** -0.5
        nn.init.uniform_(self.lin_dt.weight, -dt_init_std, dt_init_std)
        dt_bias = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        with torch.no_grad():
            self.lin_dt.bias.copy_(dt_bias)

        self.D = nn.Parameter(torch.ones(d_inner))  # skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective SSM forward pass (simplified recurrent mode).

        Args:
            x: ``(B, T, d_inner)`` input.

        Returns:
            ``(B, T, d_inner)`` output.
        """
        B, T, D = x.shape
        N = self.d_state

        # Input-dependent parameters
        dt = F.softplus(self.lin_dt(x))    # (B, T, D) — positive step sizes
        B_ = self.lin_B(x)                 # (B, T, N)
        C  = self.lin_C(x)                 # (B, T, N)
        A  = -self.log_A.exp()             # (D, N) — negative for stability

        # Discretise: Ā[t] = exp(Δ[t] * A), B̄[t] = Δ[t] * B[t]
        # shape: (B, T, D, N)
        dt_A = torch.einsum("btd,dn->btdn", dt, A)   # (B, T, D, N)
        A_bar = torch.exp(dt_A)                        # (B, T, D, N)
        B_bar = torch.einsum("btd,btn->btdn", dt, B_) # (B, T, D, N)

        # Recurrent scan: h_t = Ā_t h_{t-1} + B̄_t x_t
        h = torch.zeros(B, D, N, device=x.device)
        outputs = []
        for t in range(T):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t, :, None]  # (B, D, N)
            # y_t = C_t * h + D * x_t
            y_t = (h * C[:, t, None, :]).sum(-1) + self.D * x[:, t]  # (B, D)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)   # (B, T, D)


class MambaBlock(nn.Module):
    """
    Full Mamba block (Gu & Dao, 2023).

    Architecture:
      x → expand → [z branch: SiLU gate]
               ↘ [x branch: conv1d → SiLU → SelectiveSSM]
      y = SSM_output * SiLU(z)
      y → project_out → residual

    Args:
        cfg: :class:`SSMConfig`.

    Example::

        block = MambaBlock(SSMConfig(d_model=64, d_state=16, expand=2))
        x     = torch.randn(2, 32, 64)  # (B, T, d_model)
        y     = block(x)                 # (B, T, d_model)
    """

    def __init__(self, cfg: SSMConfig) -> None:
        super().__init__()
        self.cfg    = cfg
        d_in  = cfg.d_model
        d_i   = cfg.d_inner   # d_model * expand

        self.norm     = nn.LayerNorm(d_in)
        self.in_proj  = nn.Linear(d_in, 2 * d_i, bias=False)  # x + z
        self.conv1d   = nn.Conv1d(d_i, d_i, kernel_size=cfg.d_conv,
                                   padding=cfg.d_conv - 1,
                                   groups=d_i, bias=True)
        self.ssm      = SelectiveSSM(d_i, cfg.d_state,
                                      cfg.dt_min, cfg.dt_max)
        self.out_proj = nn.Linear(d_i, d_in, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mamba block forward pass.

        Args:
            x: ``(B, T, d_model)`` input.

        Returns:
            ``(B, T, d_model)`` output (residual NOT added here).
        """
        residual = x
        x        = self.norm(x)          # pre-norm
        B, T, _  = x.shape

        # Expand to d_inner, split into x and z branches
        xz  = self.in_proj(x)            # (B, T, 2*d_inner)
        x_, z = xz.chunk(2, dim=-1)      # each (B, T, d_inner)

        # Local conv (for local context)
        x_ = x_.transpose(1, 2)          # (B, d_inner, T)
        x_ = self.conv1d(x_)[..., :T]    # causal conv
        x_ = x_.transpose(1, 2)          # (B, T, d_inner)
        x_ = F.silu(x_)

        # Selective SSM
        y = self.ssm(x_)                 # (B, T, d_inner)

        # Gate with z branch
        y = y * F.silu(z)                # (B, T, d_inner)

        # Project back
        y = self.out_proj(y)             # (B, T, d_model)
        return y + residual


class MambaLM(nn.Module):
    """
    Mamba Language Model: stack of MambaBlocks + embedding + LM head.

    Replaces Transformer attention with linear-time SSM!
    Achieves similar perplexity to Transformers on language tasks
    with O(T) inference time and O(N) memory.

    Args:
        vocab_size: Vocabulary size.
        cfg:        :class:`SSMConfig` (d_model, d_state, expand, n_layers).
        n_layers:   Number of Mamba blocks.

    Example::

        lm    = MambaLM(vocab_size=256, cfg=SSMConfig(d_model=64), n_layers=4)
        ids   = torch.randint(0, 256, (2, 32))
        logits = lm(ids)   # (2, 32, 256) — O(T) vs O(T²) for Transformer!
    """

    def __init__(
        self,
        vocab_size: int,
        cfg:        SSMConfig,
        n_layers:   int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.layers    = nn.ModuleList([MambaBlock(cfg) for _ in range(n_layers)])
        self.norm      = nn.LayerNorm(cfg.d_model)
        self.lm_head   = nn.Linear(cfg.d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: ``(B, T)`` token IDs.

        Returns:
            ``(B, T, vocab_size)`` logits.
        """
        x = self.embedding(input_ids)   # (B, T, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)          # (B, T, vocab_size)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
