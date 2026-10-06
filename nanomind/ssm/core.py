"""
nanomind/ssm/core.py — Core Structured State Space Model (S4).

## State Space Models (SSMs)

SSMs are the mathematical foundation: a continuous-time recurrence.

Continuous-time system:
  h'(t) = A h(t) + B x(t)      (state update)
  y(t)  = C h(t) + D x(t)      (output)

Where:
  x(t) ∈ ℝ^d_in    — input signal
  h(t) ∈ ℝ^N       — hidden state (N-dimensional)
  y(t) ∈ ℝ^d_out   — output signal
  A ∈ ℝ^{N×N}      — state transition (dynamics matrix)
  B ∈ ℝ^{N×d_in}   — input projection
  C ∈ ℝ^{d_out×N}  — output projection
  D ∈ ℝ^{d_out}    — skip connection

## Discretisation (ZOH)

To process discrete token sequences, discretise with step size Δ:

  Ā = exp(Δ A)                  (matrix exponential)
  B̄ = (A)^{-1}(Ā - I) B       (ZOH discretisation)

Discrete recurrence:
  h_t = Ā h_{t-1} + B̄ x_t
  y_t = C h_t + D x_t

## Computational Modes

Two equivalent computations:

1. Recurrent mode (inference, O(N) per step):
   h_t = Ā h_{t-1} + B̄ x_t
   Perfect for autoregressive generation — O(1) step, O(N) memory

2. Convolutional mode (training, O(T log T)):
   The SSM kernel K = (C B̄, C Ā B̄, C Ā² B̄, ..., C Ā^{T-1} B̄)
   y = conv(x, K)  via FFT
   Parallel over sequence — like a causal conv!

## Why SSMs > Transformers for Long Sequences?

Transformer: O(T²) attention, O(T) memory
SSM:         O(T log T) training, O(N) inference, O(N) memory

References:
  Gu et al. (2021) S4: https://arxiv.org/abs/2111.00396
  Gu & Dao (2023) Mamba: https://arxiv.org/abs/2312.00752
  Smith et al. (2022) S5: https://arxiv.org/abs/2208.04933
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SSMConfig:
    """Configuration for an SSM layer."""
    d_model:  int   = 64    # model dimension
    d_state:  int   = 16    # SSM state dimension N
    d_conv:   int   = 4     # local conv width (Mamba)
    expand:   int   = 2     # inner expand factor (Mamba uses 2)
    dt_min:   float = 0.001 # min time step Δ
    dt_max:   float = 0.1   # max time step Δ
    dt_init:  str   = "random"
    dropout:  float = 0.0

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand


def make_hippo_matrix(N: int) -> torch.Tensor:
    """
    Construct the HiPPO-LegS matrix A (Gu et al., 2020).

    The HiPPO matrix provides theoretically-grounded initialisation
    for the SSM dynamics, encoding polynomial projections of input history.

    A[n,k] = -(2n+1)^{1/2}(2k+1)^{1/2}  if n > k
              n+1                           if n == k
              0                             if n < k

    Args:
        N: State dimension.

    Returns:
        ``(N, N)`` HiPPO-LegS matrix.
    """
    A   = torch.zeros(N, N)
    ns  = torch.arange(N, dtype=torch.float32)
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = -((2*n+1)**0.5) * ((2*k+1)**0.5)
            elif n == k:
                A[n, k] = n + 1
    return A


class DiscretizedSSM(nn.Module):
    """
    Single-channel discretised SSM (S4-style).

    Uses the Zero-Order Hold discretisation:
      Ā = exp(Δ A)
      B̄ = (A)^{-1}(Ā - I) B

    Supports both recurrent and convolutional modes.

    Args:
        d_state: SSM state dimension N.
        dt_min:  Minimum time step.
        dt_max:  Maximum time step.

    Example::

        ssm = DiscretizedSSM(d_state=16)
        x   = torch.randn(2, 32, 1)  # (B, T, 1)
        y   = ssm.forward_conv(x)    # (B, T, 1) — parallel
        h, y_last = ssm.forward_recurrent(x, h0=None)  # — sequential
    """

    def __init__(
        self,
        d_state: int   = 16,
        dt_min:  float = 0.001,
        dt_max:  float = 0.1,
    ) -> None:
        super().__init__()
        self.d_state = d_state

        # A: use diagonal approximation of HiPPO for efficiency
        # log_A_real ensures A has negative real parts (stable)
        self.log_A_real = nn.Parameter(
            torch.log(0.5 * torch.ones(d_state))
        )
        self.A_imag = nn.Parameter(
            math.pi * torch.arange(d_state).float()
        )

        # B and C are complex-valued (diagonal SSM uses complex diag A)
        self.B = nn.Parameter(torch.randn(d_state, 2) * 0.1)  # (N, 2) complex
        self.C = nn.Parameter(torch.randn(d_state, 2) * 0.1)  # (N, 2) complex
        self.D = nn.Parameter(torch.ones(1))                    # skip connection

        # Log time step Δ (scalar, learnable)
        log_dt = torch.rand(1) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    @property
    def A(self) -> torch.Tensor:
        """Complex diagonal A: a + iω"""
        return -self.log_A_real.exp() + 1j * self.A_imag

    @property
    def dt(self) -> torch.Tensor:
        return self.log_dt.exp()

    def _discretise(self) -> tuple[torch.Tensor, torch.Tensor]:
        """ZOH discretisation: (Ā, B̄)."""
        dt = self.dt
        A  = self.A                        # (N,) complex
        B  = self.B[..., 0] + 1j * self.B[..., 1]  # (N,) complex
        A_bar = torch.exp(dt * A)          # (N,) complex
        B_bar = (A_bar - 1) / A * B        # (N,) complex
        return A_bar, B_bar

    def _kernel(self, T: int) -> torch.Tensor:
        """Compute SSM convolution kernel of length T."""
        A_bar, B_bar = self._discretise()
        C = self.C[..., 0] + 1j * self.C[..., 1]
        # K[t] = C * A^t * B
        powers = torch.arange(T, device=A_bar.device).unsqueeze(-1)  # (T, 1)
        A_pow  = A_bar.unsqueeze(0) ** powers   # (T, N) complex
        K      = (C.unsqueeze(0) * A_pow * B_bar.unsqueeze(0)).sum(-1).real  # (T,) real
        return K

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convolutional (parallel) forward — O(T log T) via FFT.

        Args:
            x: ``(B, T, 1)`` input signal.

        Returns:
            ``(B, T, 1)`` output.
        """
        B, T, _ = x.shape
        K   = self._kernel(T)              # (T,) SSM kernel
        x_s = x.squeeze(-1)               # (B, T)
        # Causal convolution via FFT
        fft_size = 2 * T
        K_f = torch.fft.rfft(K,  n=fft_size)
        x_f = torch.fft.rfft(x_s, n=fft_size)
        y   = torch.fft.irfft(K_f * x_f, n=fft_size)[:, :T]  # (B, T)
        y   = y + self.D * x_s
        return y.unsqueeze(-1)             # (B, T, 1)

    def forward_recurrent(
        self,
        x:  torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent (sequential) forward — O(N) per step.

        Args:
            x:  ``(B, T, 1)`` input.
            h0: ``(B, N)`` initial hidden state (complex).

        Returns:
            (final_h, output): h is ``(B, N)`` complex, output is ``(B, T, 1)``.
        """
        B, T, _ = x.shape
        A_bar, B_bar = self._discretise()
        C = self.C[..., 0] + 1j * self.C[..., 1]

        h = h0 if h0 is not None else torch.zeros(B, self.d_state,
                                                    dtype=torch.complex64,
                                                    device=x.device)
        outputs = []
        for t in range(T):
            x_t = x[:, t, 0].to(torch.complex64)       # (B,)
            h   = A_bar * h + B_bar * x_t.unsqueeze(-1) # (B, N)
            y_t = (C * h).sum(-1).real + self.D * x[:, t, 0]  # (B,)
            outputs.append(y_t)

        output = torch.stack(outputs, dim=1).unsqueeze(-1)  # (B, T, 1)
        return h, output
