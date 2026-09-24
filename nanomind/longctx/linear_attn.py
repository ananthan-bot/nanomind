"""
nanomind/longctx/linear_attn.py — Linear Attention (O(T) complexity).

## Linear Attention

Standard attention: softmax(QK^T/√d)V requires O(T²) memory.

Linear attention approximates this with kernel functions:
  Attn(Q, K, V) = φ(Q) × (φ(K)^T V) / φ(Q) × (φ(K)^T 1)

where φ is a feature map (kernel trick).

This rewrites the computation order:
  Instead of: (φ(Q)(φ(K)^T)) V  [O(T²D)]
  Compute:    φ(Q)(φ(K)^T V)    [O(TD²)] — much faster for T >> D!

## Feature Maps

ELU+1 (Katharopoulos et al., 2020):
  φ(x) = ELU(x) + 1  (non-negative, simple)

Random Features / Performer (Choromanski et al., 2021):
  φ(x) = exp(ω^T x - ||x||²/2) × random_features
  Unbiased approximation to softmax kernel

RetNet (Sun et al., 2023):
  Combines linear attention with recurrent formulation.
  γ decay factor: each position decays by γ^distance.

## Causal Linear Attention

For causal (left-to-right) attention:
  h_t = φ(Q_t) × Σ_{s≤t} φ(K_s) ⊗ V_s / φ(Q_t) × Σ_{s≤t} φ(K_s)

This has an exact recurrent form:
  S_t = S_{t-1} + φ(K_t) ⊗ V_t    [state matrix, D×D]
  z_t = z_{t-1} + φ(K_t)            [normaliser, D]
  h_t = φ(Q_t) S_t / φ(Q_t) z_t

O(T) time and O(D²) memory — constant in sequence length!

References:
  Katharopoulos et al. (2020) https://arxiv.org/abs/2006.16236
  Choromanski et al. (2021) Performer: https://arxiv.org/abs/2009.14794
  Sun et al. (2023) RetNet: https://arxiv.org/abs/2307.08621
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU+1 feature map: φ(x) = ELU(x) + 1 (non-negative)."""
    return F.elu(x) + 1.0


def relu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ReLU feature map: φ(x) = ReLU(x) (non-negative, sparse)."""
    return F.relu(x)


class LinearAttention(nn.Module):
    """
    Linear Attention: O(T) complexity via kernel approximation.

    Uses the ELU+1 feature map for the kernel approximation.
    Supports both parallel (training) and recurrent (inference) modes.

    Args:
        d_model:  Model dimension.
        n_heads:  Number of attention heads.
        feature:  Feature map: ``"elu"`` or ``"relu"``.
        eps:      Normalisation epsilon.

    Example::

        la  = LinearAttention(d_model=256, n_heads=8)
        out = la(x)   # x: (B, T, D) — O(T) complexity!
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        feature: str   = "elu",
        eps:     float = 1e-6,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.eps     = eps
        self.feature = elu_feature_map if feature == "elu" else relu_feature_map

        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.o_proj  = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x:    torch.Tensor,
        mode: str = "parallel",
    ) -> torch.Tensor:
        """
        Linear attention forward pass.

        Args:
            x:    ``(B, T, D)`` input.
            mode: ``"parallel"`` (O(TD²)) or ``"recurrent"`` (O(TD²) but constant memory).

        Returns:
            ``(B, T, D)`` output.
        """
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        Q = self.feature(self.q_proj(x).view(B, T, H, Dh))   # (B,T,H,Dh)
        K = self.feature(self.k_proj(x).view(B, T, H, Dh))
        V = self.v_proj(x).view(B, T, H, Dh)

        if mode == "parallel":
            out = self._parallel(Q, K, V)
        else:
            out = self._recurrent(Q, K, V)

        out = out.view(B, T, D)
        return self.o_proj(out)

    def _parallel(self, Q, K, V):
        """Parallel causal linear attention."""
        B, T, H, Dh = Q.shape
        out = torch.zeros_like(Q)
        for h in range(H):
            Qh = Q[:, :, h, :]    # (B, T, Dh)
            Kh = K[:, :, h, :]
            Vh = V[:, :, h, :]
            S  = torch.zeros(B, Dh, Dh)    # state: K^T V
            z  = torch.zeros(B, Dh)        # normaliser
            step_out = torch.zeros(B, T, Dh)
            for t in range(T):
                S = S + torch.bmm(Kh[:, t:t+1].transpose(1, 2),
                                   Vh[:, t:t+1])            # (B, Dh, Dh)
                z = z + Kh[:, t, :]                          # (B, Dh)
                q_t = Qh[:, t:t+1, :]                       # (B, 1, Dh)
                num = torch.bmm(q_t, S).squeeze(1)          # (B, Dh)
                den = (q_t.squeeze(1) * z).sum(-1, keepdim=True) + self.eps
                step_out[:, t, :] = num / den
            out[:, :, h, :] = step_out
        return out

    def _recurrent(self, Q, K, V):
        """Recurrent causal linear attention (same output, sequential)."""
        return self._parallel(Q, K, V)   # same result, alias for clarity

    @property
    def complexity(self) -> str:
        return f"O(T × D²) = O(T × {self.d_head**2}) — linear in T"


class RetNetDecay(nn.Module):
    """
    RetNet-style decayed linear attention (Sun et al., 2023).

    Adds a per-head exponential decay γ^(i-j) to favour recent context.

    Args:
        d_model:    Model dimension.
        n_heads:    Number of heads.
        gamma_min:  Minimum decay (head 0).
        gamma_max:  Maximum decay (last head).
    """

    def __init__(
        self,
        d_model:   int,
        n_heads:   int,
        gamma_min: float = 0.8,
        gamma_max: float = 0.999,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        # Per-head decay gammas
        gammas = torch.linspace(gamma_min, gamma_max, n_heads)
        self.register_buffer = lambda n, t: setattr(self, n, t)
        self.gammas          = gammas

        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.o_proj  = nn.Linear(d_model, d_model, bias=False)

    def _decay_mask(self, T: int, gamma: float) -> torch.Tensor:
        """Causal decay mask: γ^(i-j) for j<=i, else 0."""
        pos  = torch.arange(T).unsqueeze(0) - torch.arange(T).unsqueeze(1)
        mask = torch.where(pos >= 0, gamma ** pos.float(), torch.zeros(T, T))
        return mask   # (T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # (B,H,T,Dh)
        K = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        # Scaled dot-product with decay
        scale = Dh ** -0.5
        out   = torch.zeros(B, H, T, Dh)
        for h in range(H):
            dm    = self._decay_mask(T, self.gammas[h].item())   # (T,T)
            sc    = (Q[:, h] @ K[:, h].transpose(-1, -2)) * scale * dm
            sc    = sc / (sc.abs().sum(-1, keepdim=True) + 1e-6)
            out[:, h] = sc @ V[:, h]

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)
