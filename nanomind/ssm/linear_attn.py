"""
nanomind/ssm/linear_attn.py — Linear Attention and RWKV-style recurrent models.

## Linear Attention (Katharopoulos et al., 2020)

Standard softmax attention:
  Attention(Q, K, V) = softmax(QK^T / √d) V   — O(T²)

Linear attention: replace softmax with kernel function φ:
  Attention(Q, K, V) = φ(Q) (φ(K)^T V) / φ(Q) φ(K)^T 1   — O(T)!

Key insight: (φ(K)^T V) can be computed incrementally!
  S_t = S_{t-1} + φ(k_t) v_t^T    (numerator accumulator)
  z_t = z_{t-1} + φ(k_t)           (denominator accumulator)
  y_t = φ(q_t) S_t / φ(q_t) z_t

This is a linear RNN! Same expressivity, O(1) per step.

## RWKV (Peng et al., 2023)

RWKV = Receptance Weighted Key Value
Combines RNN efficiency with Transformer parallelism.

RWKV recurrence:
  w_t = exp(-exp(w))        — exponential decay weights
  u   = bonus for current t
  wkv_t = (Σ_{i<t} exp(w(t-i-1)+k_i) v_i + exp(u+k_t) v_t) /
          (Σ_{i<t} exp(w(t-i-1)+k_i)      + exp(u+k_t))

WKV is a weighted average of values, with exponentially decaying weights.

In matrix form (parallel training):
  Y = diag(r) × WKV  — pointwise rescaling by "receptance" r

## Retention (Microsoft, 2023)

Retentive Networks combine benefits of RNN + Transformer:
  Retention(X) = (QK^T ⊙ D) V  where D_{ij} = γ^{i-j} if i>=j else 0

The decay mask D makes it causal + recurrent.

References:
  Katharopoulos et al. (2020) "Transformers are RNNs"
  https://arxiv.org/abs/2006.16236
  Peng et al. (2023) RWKV: https://arxiv.org/abs/2305.13048
  Sun et al. (2023) Retentive: https://arxiv.org/abs/2307.08621
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class LinearAttnConfig:
    """Configuration for linear attention models."""
    d_model:   int   = 64
    n_heads:   int   = 4
    d_head:    int   = 16      # d_model // n_heads
    dropout:   float = 0.1
    eps:       float = 1e-6    # numerical stability


class LinearAttention(nn.Module):
    """
    Causal linear attention: O(T) training, O(1) recurrent inference.

    Uses ELU+1 as the kernel function φ(x) = elu(x) + 1 > 0.

    Args:
        cfg: :class:`LinearAttnConfig`.

    Example::

        attn = LinearAttention(LinearAttnConfig(d_model=64, n_heads=4))
        x    = torch.randn(2, 32, 64)
        y    = attn(x)   # (2, 32, 64) — same shape, O(T) computation
    """

    def __init__(self, cfg: LinearAttnConfig) -> None:
        super().__init__()
        self.cfg   = cfg
        self.heads = cfg.n_heads
        self.d_h   = cfg.d_head

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.out    = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

    @staticmethod
    def _kernel(x: torch.Tensor) -> torch.Tensor:
        """φ(x) = elu(x) + 1 (positive, bounded below)."""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Causal linear attention forward pass.

        Args:
            x: ``(B, T, d_model)``.

        Returns:
            ``(B, T, d_model)``.
        """
        B, T, _ = x.shape
        H, d    = self.heads, self.d_h

        Q = self._kernel(self.q_proj(x)).view(B, T, H, d)
        K = self._kernel(self.k_proj(x)).view(B, T, H, d)
        V = self.v_proj(x).view(B, T, H, d)

        # Causal linear attention via cumulative sum
        # S_t = Σ_{i<=t} K_i V_i^T  (numerator)
        # z_t = Σ_{i<=t} K_i         (denominator)
        # y_t = Q_t S_t / (Q_t z_t)
        KV = torch.einsum("bthd,bthe->bthde", K, V)  # (B, T, H, d, d)
        S  = KV.cumsum(dim=1)                          # (B, T, H, d, d)
        z  = K.cumsum(dim=1)                           # (B, T, H, d)

        # Numerator: Q S
        y_num = torch.einsum("bthd,bthde->bthe", Q, S)  # (B, T, H, d)
        # Denominator: Q z
        y_den = (Q * z).sum(dim=-1, keepdim=True).clamp(min=self.cfg.eps)  # (B, T, H, 1)

        y = (y_num / y_den).view(B, T, H * d)   # (B, T, d_model)
        return self.out(y)

    def recurrent_step(
        self,
        x_t: torch.Tensor,
        S:   torch.Tensor,
        z:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single recurrent step (O(1) inference).

        Args:
            x_t: ``(B, d_model)`` current input.
            S:   ``(B, H, d, d)`` state numerator.
            z:   ``(B, H, d)`` state denominator.

        Returns:
            (y_t, S_new, z_new).
        """
        B  = x_t.shape[0]
        H, d = self.heads, self.d_h

        q = self._kernel(self.q_proj(x_t)).view(B, H, d)
        k = self._kernel(self.k_proj(x_t)).view(B, H, d)
        v = self.v_proj(x_t).view(B, H, d)

        S_new = S + torch.einsum("bhd,bhe->bhde", k, v)
        z_new = z + k

        y_num = torch.einsum("bhd,bhde->bhe", q, S_new)
        y_den = (q * z_new).sum(-1, keepdim=True).clamp(min=self.cfg.eps)
        y_t   = (y_num / y_den).view(B, H * d)
        return self.out(y_t), S_new, z_new


class RetentiveLayer(nn.Module):
    """
    Retention layer (Retentive Networks, Sun et al., 2023).

    Uses a geometric decay mask D where D[i,j] = γ^{i-j} if i>=j.
    This makes retention a linear-time, recurrent-compatible attention.

    Args:
        d_model:  Model dimension.
        n_heads:  Number of retention heads.
        gamma:    Decay rate per head (default: computed from n_heads).

    Example::

        layer  = RetentiveLayer(d_model=64, n_heads=4)
        x      = torch.randn(2, 32, 64)
        y      = layer(x)   # (2, 32, 64)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        gamma:   list[float] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        # Default gammas: evenly spaced in [1 - 2^(-5), 1 - 2^(-9)]
        if gamma is None:
            gamma = [1 - 2 ** (-5 - i) for i in range(n_heads)]
        self.register_buffer("gamma", torch.tensor(gamma))

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out    = nn.Linear(d_model, d_model, bias=False)
        self.norm   = nn.GroupNorm(n_heads, d_model)

    def _decay_mask(self, T: int, device) -> torch.Tensor:
        """Build causal decay mask D: (H, T, T) where D[h,i,j] = γ_h^{i-j}."""
        idx = torch.arange(T, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)   # (T, T) — i-j
        mask = self.gamma.view(-1, 1, 1) ** diff.unsqueeze(0)
        mask = mask * (diff >= 0).float().unsqueeze(0)   # causal: 0 if i < j
        return mask  # (H, T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retention forward (parallel mode).

        Args:
            x: ``(B, T, d_model)``.

        Returns:
            ``(B, T, d_model)``.
        """
        B, T, _ = x.shape
        H, d    = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)  # (B, H, T, d)
        K = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        # Decay mask
        D = self._decay_mask(T, x.device)   # (H, T, T)

        # Retention scores: Q K^T ⊙ D / sqrt(d)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d)   # (B, H, T, T)
        scores = scores * D.unsqueeze(0)                      # apply decay
        y      = scores @ V                                   # (B, H, T, d)

        y = y.transpose(1, 2).reshape(B, T, H * d)  # (B, T, d_model)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)   # GroupNorm per head
        return self.out(y)
