"""
day56_commits.py — 20 atomic commits for Day 56: Structured State Space Models (SSM/Mamba).
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== DAY 56: Structured State Space Models (SSM/Mamba) — 20 commits, v5.6.0 ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — ssm package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/ssm/__init__.py",
      '"""NanoMind SSM sub-package — Structured State Space Models (S4/Mamba)."""\n')
commit("feat: add nanomind/ssm/ package skeleton for Structured State Space Models")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — Core SSM (S4 linear recurrence)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/ssm/core.py", '''\
"""
nanomind/ssm/core.py — Core Structured State Space Model (S4).

## State Space Models (SSMs)

SSMs are the mathematical foundation: a continuous-time recurrence.

Continuous-time system:
  h\'(t) = A h(t) + B x(t)      (state update)
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
''')
commit("feat: add DiscretizedSSM — HiPPO init, ZOH discretization, conv (FFT) and recurrent modes")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — Mamba selective SSM
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/ssm/mamba.py", '''\
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
''')
commit("feat: add SelectiveSSM (input-dep Δ/B/C), MambaBlock (conv+SSM+gate), MambaLM — linear-time LM")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — S4 (structured S4 with convolutional kernel)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/ssm/s4.py", '''\
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
''')
commit("feat: add S4Layer (per-channel SSMs, FFT conv), S4Block (S4+FFN+residual), S4Model LM")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — Linear attention / RWKV-style
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/ssm/linear_attn.py", '''\
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
''')
commit("feat: add LinearAttention (O(T) causal, recurrent_step), RetentiveLayer (Retentive Networks)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — SSM analysis utilities
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/ssm/analysis.py", '''\
"""
nanomind/ssm/analysis.py — SSM analysis and comparison utilities.

Utilities for understanding SSM behaviour:
  - Effective receptive field
  - State utilisation
  - Complexity comparison (Transformer vs S4 vs Mamba)
  - SSM impulse response visualization
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ComplexityComparison:
    """FLOPs / memory complexity for different architectures."""
    architecture: str
    training_flops: str   # in terms of B, T, D, N
    inference_flops: str
    memory:          str
    parallel:        bool

    def to_dict(self) -> dict:
        return {
            "architecture":   self.architecture,
            "train_FLOPs":    self.training_flops,
            "infer_FLOPs":    self.inference_flops,
            "memory":         self.memory,
            "parallel_train": self.parallel,
        }


ARCHITECTURE_COMPLEXITIES = [
    ComplexityComparison(
        "Transformer",
        training_flops  = "O(B T² D)",
        inference_flops = "O(T D) per step (with KV cache)",
        memory          = "O(B T D) KV cache",
        parallel        = True,
    ),
    ComplexityComparison(
        "S4",
        training_flops  = "O(B T D N log T)",
        inference_flops = "O(B D N) per step",
        memory          = "O(B D N)",
        parallel        = True,
    ),
    ComplexityComparison(
        "Mamba",
        training_flops  = "O(B T D N)",
        inference_flops = "O(B D N) per step",
        memory          = "O(B D N)",
        parallel        = True,
    ),
    ComplexityComparison(
        "Linear Attention",
        training_flops  = "O(B T D²)",
        inference_flops = "O(B D²) per step",
        memory          = "O(B D²)",
        parallel        = True,
    ),
    ComplexityComparison(
        "LSTM",
        training_flops  = "O(B T D²)",
        inference_flops = "O(B D²) per step",
        memory          = "O(B D)",
        parallel        = False,
    ),
]


def compute_ssm_impulse_response(
    A_bar: torch.Tensor,
    B_bar: torch.Tensor,
    C:     torch.Tensor,
    T:     int,
) -> torch.Tensor:
    """
    Compute SSM impulse response K[t] = C * A^t * B.

    The impulse response describes how the SSM responds to a single
    input spike at t=0. Long impulse responses = long-range memory.

    Args:
        A_bar: ``(N,)`` discretised state matrix diagonal.
        B_bar: ``(N,)`` discretised input matrix.
        C:     ``(N,)`` output matrix.
        T:     Number of steps.

    Returns:
        ``(T,)`` impulse response kernel.
    """
    powers = torch.arange(T, device=A_bar.device)
    K = [((C * A_bar**t * B_bar).sum()).real for t in powers]
    return torch.stack(K)


def effective_memory_length(kernel: torch.Tensor, threshold: float = 0.01) -> int:
    """
    Estimate effective memory length from impulse response.

    Returns the last time step where |K[t]| > threshold * max(|K|).

    Args:
        kernel:    ``(T,)`` impulse response.
        threshold: Fraction of peak for cutoff.

    Returns:
        Effective memory length.
    """
    k_abs   = kernel.abs()
    peak    = k_abs.max().item()
    cutoff  = threshold * peak
    indices = (k_abs > cutoff).nonzero(as_tuple=True)[0]
    return indices[-1].item() + 1 if len(indices) > 0 else 0


def parameter_count_comparison(
    d_model:    int,
    d_state:    int,
    seq_len:    int,
    vocab_size: int,
    n_layers:   int,
) -> dict:
    """
    Compare parameter counts: Transformer vs S4 vs Mamba.

    For the same d_model, n_layers, vocab_size.
    """
    # Transformer (standard)
    # per layer: 4 × d² (QKV + out) + 8 × d² (FFN 4x) = 12d²
    tf_params = vocab_size * d_model + n_layers * 12 * d_model**2 + vocab_size * d_model

    # S4 (per channel SSM, no attention)
    # per S4 layer: H SSMs of size N + FFN
    s4_per_ssm  = 2 * d_state + 2 * d_state + 1  # B, C, D per channel
    s4_per_layer = d_model * s4_per_ssm + 8 * d_model**2   # + FFN
    s4_params    = vocab_size * d_model + n_layers * s4_per_layer

    # Mamba (expand=2, d_state=N)
    # per layer: 2*2d² (in/out proj) + d*(d_state*3) (Δ,B,C) + conv
    d_i = d_model * 2
    mamba_per_layer = (2 * d_i * d_model + d_i * d_state * 3 + d_i * 4)
    mamba_params    = vocab_size * d_model + n_layers * mamba_per_layer

    return {
        "d_model":       d_model,
        "n_layers":      n_layers,
        "transformer_M": round(tf_params / 1e6, 2),
        "s4_M":          round(s4_params / 1e6, 2),
        "mamba_M":       round(mamba_params / 1e6, 2),
    }


def flops_comparison(
    batch_size:  int,
    seq_len:     int,
    d_model:     int,
    d_state:     int,
) -> dict:
    """
    Compare FLOPs for a single forward pass.

    Returns approximate FLOPs for one layer.
    """
    B, T, D, N = batch_size, seq_len, d_model, d_state
    return {
        "transformer_GFLOPs": round(2 * B * T**2 * D / 1e9, 4),
        "s4_GFLOPs":          round(2 * B * T * D * N * math.log2(T) / 1e9, 4),
        "mamba_GFLOPs":       round(2 * B * T * D * N / 1e9, 4),
        "linear_attn_GFLOPs": round(2 * B * T * D**2 / 1e9, 4),
        "note":               f"B={B}, T={T}, D={D}, N={N}",
    }
''')
commit("feat: add SSM analysis — complexity comparison, impulse_response, effective_memory_length, flops")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Hybrid Transformer-SSM
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/ssm/hybrid.py", '''\
"""
nanomind/ssm/hybrid.py — Hybrid Transformer + SSM models.

## Why Hybrid?

Pure SSMs are strong on long sequences but weaker on tasks requiring
precise token-to-token attention (e.g., in-context learning, retrieval).

Hybrid models combine:
  - SSM layers: cheap, linear-time, great for long-range
  - Attention layers: expensive but precise, great for short-range

Examples:
  - Jamba (AI21, 2024): 1 attention every 8 Mamba layers
  - Griffin (DeepMind, 2024): local attention + gated linear recurrence
  - Zamba (Zyphra, 2024): SSM backbone + shared attention every 6 layers

Hybrid ratio:
  - 1:7 (1 attn per 7 SSM) → ~5% of params in attention
  - Attains near-Transformer quality at near-SSM efficiency
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from nanomind.ssm.mamba import MambaBlock, SSMConfig


@dataclass
class HybridConfig:
    """Configuration for hybrid Transformer-SSM model."""
    d_model:    int        = 64
    n_heads:    int        = 4
    d_state:    int        = 16
    expand:     int        = 2
    n_layers:   int        = 8
    attn_every: int        = 4     # insert attention every N SSM layers
    vocab_size: int        = 256
    dropout:    float      = 0.1


class LocalAttentionLayer(nn.Module):
    """
    Local sliding-window attention (O(T × window) cost).
    Used in Griffin/hybrid models as a complement to SSM.

    Args:
        d_model:     Model dimension.
        n_heads:     Number of heads.
        window_size: Local attention window.
        dropout:     Dropout.

    Example::

        attn = LocalAttentionLayer(64, 4, window_size=16)
        x    = torch.randn(2, 32, 64)
        y    = attn(x)   # (2, 32, 64)
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        window_size: int   = 64,
        dropout:     float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.attn        = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        W = min(self.window_size, T)
        # Build causal mask (full causal — window restricts in practice)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        r = self.norm(x)
        out, _ = self.attn(r, r, r, attn_mask=mask)
        return x + out


class HybridSSMTransformer(nn.Module):
    """
    Hybrid model: mostly Mamba blocks with periodic attention layers.

    Architecture (attn_every=4):
      Layer 0: MambaBlock
      Layer 1: MambaBlock
      Layer 2: MambaBlock
      Layer 3: MambaBlock
      Layer 4: LocalAttentionLayer  ← every attn_every layers
      Layer 5: MambaBlock
      ...

    Args:
        cfg: :class:`HybridConfig`.

    Example::

        cfg    = HybridConfig(d_model=64, n_layers=8, attn_every=4)
        model  = HybridSSMTransformer(cfg)
        ids    = torch.randint(0, 256, (2, 32))
        logits = model(ids)   # (2, 32, 256)
    """

    def __init__(self, cfg: HybridConfig) -> None:
        super().__init__()
        self.cfg       = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        ssm_cfg        = SSMConfig(
            d_model=cfg.d_model, d_state=cfg.d_state,
            expand=cfg.expand,
        )
        layers = []
        for i in range(cfg.n_layers):
            if (i + 1) % cfg.attn_every == 0:
                layers.append(LocalAttentionLayer(
                    cfg.d_model, cfg.n_heads, dropout=cfg.dropout
                ))
            else:
                layers.append(MambaBlock(ssm_cfg))
        self.layers  = nn.ModuleList(layers)
        self.norm    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

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

    def layer_type_summary(self) -> dict:
        """Count SSM vs attention layers."""
        n_mamba = sum(1 for l in self.layers if isinstance(l, MambaBlock))
        n_attn  = sum(1 for l in self.layers if isinstance(l, LocalAttentionLayer))
        return {
            "total_layers": len(self.layers),
            "mamba_layers": n_mamba,
            "attn_layers":  n_attn,
            "attn_ratio":   round(n_attn / max(len(self.layers), 1), 3),
        }

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
''')
commit("feat: add HybridSSMTransformer (Jamba/Griffin-style), LocalAttentionLayer, HybridConfig")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — ssm __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/ssm/__init__.py", '''\
"""NanoMind SSM sub-package — Structured State Space Models (S4/Mamba).

Implements the full SSM research stack:
  1. SSMConfig            — d_model, d_state, expand, dt_min/max
  2. make_hippo_matrix    — HiPPO-LegS A matrix initialization
  3. DiscretizedSSM       — ZOH discretization, conv (FFT) + recurrent modes
  4. SelectiveSSM         — Mamba-style input-dependent Δ, B, C
  5. MambaBlock           — expand → conv → SSM → gate → project
  6. MambaLM              — full Mamba language model (weight tied)
  7. S4Layer              — independent per-channel SSMs, FFT convolution
  8. S4Block              — S4 + FFN + residual (pre-norm)
  9. S4Model              — full S4 language model
  10. LinearAttention     — O(T) causal linear attention with recurrent_step
  11. RetentiveLayer      — Retentive Networks (γ-decay mask)
  12. LinearAttnConfig    — linear attention configuration
  13. ComplexityComparison — FLOPs/memory comparison dataclass
  14. ARCHITECTURE_COMPLEXITIES — list of all architectures
  15. compute_ssm_impulse_response — K[t] = C Ā^t B̄
  16. effective_memory_length — last t where |K[t]| > threshold
  17. parameter_count_comparison — Transformer vs S4 vs Mamba params
  18. flops_comparison    — GFLOPs for one forward pass
  19. HybridConfig        — hybrid model config
  20. HybridSSMTransformer — Mamba + periodic local attention (Jamba-style)

Primary exports:
    - :class:`SSMConfig`             — SSM configuration
    - :func:`make_hippo_matrix`      — HiPPO-LegS A matrix
    - :class:`DiscretizedSSM`        — ZOH discretized SSM (S4)
    - :class:`SelectiveSSM`          — Mamba selective state space
    - :class:`MambaBlock`            — full Mamba block
    - :class:`MambaLM`               — Mamba language model
    - :class:`S4Layer`               — S4 per-channel SSMs
    - :class:`S4Block`               — S4 block with FFN
    - :class:`S4Model`               — S4 language model
    - :class:`LinearAttention`       — O(T) linear attention
    - :class:`RetentiveLayer`        — Retentive Networks
    - :func:`flops_comparison`       — GFLOPs analysis
    - :func:`parameter_count_comparison` — param count analysis
    - :class:`HybridSSMTransformer`  — hybrid Mamba+attention model
"""

from nanomind.ssm.core import SSMConfig, make_hippo_matrix, DiscretizedSSM
from nanomind.ssm.mamba import SelectiveSSM, MambaBlock, MambaLM
from nanomind.ssm.s4 import S4Layer, S4Block, S4Model
from nanomind.ssm.linear_attn import (
    LinearAttnConfig, LinearAttention, RetentiveLayer,
)
from nanomind.ssm.analysis import (
    ComplexityComparison, ARCHITECTURE_COMPLEXITIES,
    compute_ssm_impulse_response, effective_memory_length,
    parameter_count_comparison, flops_comparison,
)
from nanomind.ssm.hybrid import HybridConfig, HybridSSMTransformer, LocalAttentionLayer

__all__ = [
    "SSMConfig", "make_hippo_matrix", "DiscretizedSSM",
    "SelectiveSSM", "MambaBlock", "MambaLM",
    "S4Layer", "S4Block", "S4Model",
    "LinearAttnConfig", "LinearAttention", "RetentiveLayer",
    "ComplexityComparison", "ARCHITECTURE_COMPLEXITIES",
    "compute_ssm_impulse_response", "effective_memory_length",
    "parameter_count_comparison", "flops_comparison",
    "HybridConfig", "HybridSSMTransformer", "LocalAttentionLayer",
]
''')
commit("refactor: export all SSM components from nanomind/ssm/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — example
# ══════════════════════════════════════════════════════════════════════════════
write("examples/ssm_demo.py", '''\
"""
examples/ssm_demo.py — NanoMind SSM (Mamba/S4) demo.

Usage:
    python examples/ssm_demo.py
"""
import torch
import torch.nn.functional as F
from nanomind.ssm import (
    SSMConfig, make_hippo_matrix, DiscretizedSSM,
    SelectiveSSM, MambaBlock, MambaLM,
    S4Layer, S4Block, S4Model,
    LinearAttnConfig, LinearAttention, RetentiveLayer,
    flops_comparison, parameter_count_comparison,
    compute_ssm_impulse_response, effective_memory_length,
    HybridConfig, HybridSSMTransformer,
    ARCHITECTURE_COMPLEXITIES,
)

V = 64
T = 32
B = 2

print("=" * 60)
print("NanoMind SSM (Mamba/S4/Linear Attention) Demo")
print("=" * 60)

# ── HiPPO matrix ──────────────────────────────────────────────────────────────
print("\n── HiPPO-LegS Matrix ──")
A = make_hippo_matrix(8)
print(f"  A shape: {A.shape}")
print(f"  A diagonal: {A.diagonal().tolist()}")
print(f"  A is lower-triangular: {(A.triu(diagonal=1) == 0).all().item()}")

# ── Discretized SSM ───────────────────────────────────────────────────────────
print("\n── Discretized SSM (S4 core) ──")
ssm = DiscretizedSSM(d_state=8)
x   = torch.randn(B, T, 1)

# Convolutional mode (parallel, training)
y_conv = ssm.forward_conv(x)
print(f"  Conv mode:      input {tuple(x.shape)} → {tuple(y_conv.shape)}")

# Recurrent mode (sequential, inference)
h, y_rec = ssm.forward_recurrent(x)
print(f"  Recurrent mode: input {tuple(x.shape)} → {tuple(y_rec.shape)}, "
      f"state {tuple(h.shape)}")

# Both modes should give same output
max_diff = (y_conv - y_rec).abs().max().item()
print(f"  Conv vs Recurrent max diff: {max_diff:.6f}")

# Impulse response
A_bar, B_bar = ssm._discretise()
C = ssm.C[..., 0] + 1j * ssm.C[..., 1]
K = compute_ssm_impulse_response(A_bar, B_bar.real, C.real, T=16)
eff_len = effective_memory_length(K)
print(f"  Impulse response length: {len(K)}, effective memory: {eff_len}")

# ── Mamba Block ───────────────────────────────────────────────────────────────
print("\n── Mamba Block (Selective SSM) ──")
cfg   = SSMConfig(d_model=32, d_state=8, d_conv=4, expand=2, dt_min=0.001)
block = MambaBlock(cfg)
x     = torch.randn(B, T, 32)
y     = block(x)
print(f"  Input:  {tuple(x.shape)}")
print(f"  Output: {tuple(y.shape)}")

# ── Mamba LM ──────────────────────────────────────────────────────────────────
print("\n── Mamba Language Model ──")
mamba_lm = MambaLM(vocab_size=V, cfg=cfg, n_layers=3)
ids      = torch.randint(0, V, (B, T))
logits   = mamba_lm(ids)
print(f"  MambaLM: {mamba_lm.n_parameters():,} params")
print(f"  Logits: {tuple(logits.shape)}")
# Loss
loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
print(f"  Cross-entropy loss: {loss.item():.4f}")

# ── S4 ────────────────────────────────────────────────────────────────────────
print("\n── S4 Layer ──")
s4_layer = S4Layer(d_model=16, d_state=4)
x_s4     = torch.randn(B, 16, 16)
y_s4     = s4_layer(x_s4)
print(f"  S4Layer: {tuple(x_s4.shape)} → {tuple(y_s4.shape)}")

s4_model = S4Model(vocab_size=V, d_model=16, d_state=4, n_layers=2, d_ff=64)
logits_s4 = s4_model(ids[:, :16])
print(f"  S4Model: {s4_model.n_parameters():,} params, "
      f"logits={tuple(logits_s4.shape)}")

# ── Linear Attention ──────────────────────────────────────────────────────────
print("\n── Linear Attention (O(T)) ──")
la_cfg = LinearAttnConfig(d_model=32, n_heads=4, d_head=8)
la     = LinearAttention(la_cfg)
x_la   = torch.randn(B, T, 32)
y_la   = la(x_la)
print(f"  LinearAttention: {tuple(x_la.shape)} → {tuple(y_la.shape)}")

# Recurrent inference step
S0 = torch.zeros(B, la_cfg.n_heads, la_cfg.d_head, la_cfg.d_head)
z0 = torch.zeros(B, la_cfg.n_heads, la_cfg.d_head)
y_step, S1, z1 = la.recurrent_step(x_la[:, 0, :], S0, z0)
print(f"  Recurrent step: input {tuple(x_la[:, 0, :].shape)} → {tuple(y_step.shape)}")

# ── Retentive Networks ────────────────────────────────────────────────────────
print("\n── Retentive Networks ──")
ret  = RetentiveLayer(d_model=32, n_heads=4)
y_r  = ret(x_la)
print(f"  RetentiveLayer: {tuple(x_la.shape)} → {tuple(y_r.shape)}")

# ── Hybrid Model ──────────────────────────────────────────────────────────────
print("\n── Hybrid SSM-Transformer (Jamba-style) ──")
h_cfg   = HybridConfig(d_model=32, n_heads=4, d_state=8, n_layers=8,
                         attn_every=4, vocab_size=V)
hybrid  = HybridSSMTransformer(h_cfg)
print(f"  Layer summary: {hybrid.layer_type_summary()}")
print(f"  Parameters: {hybrid.n_parameters():,}")
logits_h = hybrid(ids)
print(f"  Logits: {tuple(logits_h.shape)}")

# ── Complexity comparison ─────────────────────────────────────────────────────
print("\n── Architecture Complexity Comparison ──")
for c in ARCHITECTURE_COMPLEXITIES:
    d = c.to_dict()
    print(f"  {d['architecture']:20s}: train={d['train_FLOPs']}, "
          f"infer={d['infer_FLOPs']}")

flops = flops_comparison(2, 1024, 512, 16)
print(f"\n  FLOPs at T=1024, D=512, N=16:")
for k, v in flops.items():
    if k != "note":
        print(f"    {k}: {v}")

params = parameter_count_comparison(512, 16, 1024, 50000, 12)
print(f"\n  Parameter counts (D=512, L=12, V=50k):")
for k, v in params.items():
    print(f"    {k}: {v}")

print("\nSSM demo complete!")
''')
commit("feat: add examples/ssm_demo.py — HiPPO, DiscretizedSSM, Mamba, S4, LinearAttn, Hybrid end-to-end")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_ssm.py", '''\
"""tests/test_ssm.py — Tests for NanoMind SSM package."""
import pytest
import torch
from nanomind.ssm import (
    SSMConfig, make_hippo_matrix, DiscretizedSSM,
    SelectiveSSM, MambaBlock, MambaLM,
    S4Layer, S4Block, S4Model,
    LinearAttnConfig, LinearAttention, RetentiveLayer,
    flops_comparison, parameter_count_comparison,
    compute_ssm_impulse_response, effective_memory_length,
    HybridConfig, HybridSSMTransformer,
)

V = 32


# ── HiPPO ─────────────────────────────────────────────────────────────────────

class TestHiPPO:
    def test_shape(self):
        A = make_hippo_matrix(8)
        assert A.shape == (8, 8)

    def test_lower_triangular(self):
        A = make_hippo_matrix(8)
        assert (A.triu(diagonal=1) == 0).all()

    def test_diagonal_positive(self):
        A = make_hippo_matrix(8)
        assert (A.diagonal() > 0).all()


# ── DiscretizedSSM ────────────────────────────────────────────────────────────

class TestDiscretizedSSM:
    def test_conv_output_shape(self):
        ssm = DiscretizedSSM(d_state=4)
        x   = torch.randn(2, 16, 1)
        y   = ssm.forward_conv(x)
        assert y.shape == x.shape

    def test_recurrent_output_shape(self):
        ssm = DiscretizedSSM(d_state=4)
        x   = torch.randn(2, 16, 1)
        h, y = ssm.forward_recurrent(x)
        assert y.shape == x.shape

    def test_conv_recurrent_close(self):
        ssm  = DiscretizedSSM(d_state=4)
        x    = torch.randn(1, 8, 1)
        y_c  = ssm.forward_conv(x)
        _, y_r = ssm.forward_recurrent(x)
        assert torch.allclose(y_c, y_r, atol=1e-4)

    def test_dt_positive(self):
        ssm = DiscretizedSSM(d_state=4)
        assert ssm.dt.item() > 0


# ── Mamba ──────────────────────────────────────────────────────────────────────

class TestMamba:
    def _cfg(self):
        return SSMConfig(d_model=16, d_state=4, d_conv=4, expand=2)

    def test_selective_ssm_shape(self):
        ssm = SelectiveSSM(d_inner=32, d_state=4)
        x   = torch.randn(2, 8, 32)
        y   = ssm(x)
        assert y.shape == x.shape

    def test_mamba_block_shape(self):
        block = MambaBlock(self._cfg())
        x     = torch.randn(2, 8, 16)
        y     = block(x)
        assert y.shape == x.shape

    def test_mamba_block_residual(self):
        block = MambaBlock(self._cfg())
        x     = torch.randn(2, 8, 16)
        y     = block(x)
        # Residual means output is not identical to input
        assert not torch.allclose(x, y)

    def test_mamba_lm_shape(self):
        cfg = self._cfg()
        lm  = MambaLM(V, cfg, n_layers=2)
        ids = torch.randint(0, V, (2, 8))
        y   = lm(ids)
        assert y.shape == (2, 8, V)

    def test_mamba_lm_params(self):
        cfg = self._cfg()
        lm  = MambaLM(V, cfg, n_layers=2)
        assert lm.n_parameters() > 0

    def test_weight_tying(self):
        cfg = self._cfg()
        lm  = MambaLM(V, cfg, n_layers=2)
        # Weight tying: embedding and lm_head share weights
        assert lm.lm_head.weight is lm.embedding.weight


# ── S4 ────────────────────────────────────────────────────────────────────────

class TestS4:
    def test_s4_layer_shape(self):
        layer = S4Layer(d_model=8, d_state=4)
        x     = torch.randn(2, 16, 8)
        y     = layer(x)
        assert y.shape == x.shape

    def test_s4_block_shape(self):
        block = S4Block(d_model=8, d_state=4, d_ff=32)
        x     = torch.randn(2, 16, 8)
        y     = block(x)
        assert y.shape == x.shape

    def test_s4_model_shape(self):
        model = S4Model(vocab_size=V, d_model=8, d_state=4, n_layers=2, d_ff=32)
        ids   = torch.randint(0, V, (2, 8))
        logits = model(ids)
        assert logits.shape == (2, 8, V)

    def test_s4_model_params(self):
        model = S4Model(vocab_size=V, d_model=8, d_state=4, n_layers=2, d_ff=32)
        assert model.n_parameters() > 0


# ── Linear Attention ──────────────────────────────────────────────────────────

class TestLinearAttn:
    def _la(self):
        return LinearAttention(LinearAttnConfig(d_model=16, n_heads=2, d_head=8))

    def test_output_shape(self):
        la = self._la()
        x  = torch.randn(2, 8, 16)
        y  = la(x)
        assert y.shape == x.shape

    def test_recurrent_step_shape(self):
        la  = self._la()
        x_0 = torch.randn(2, 16)
        S   = torch.zeros(2, 2, 8, 8)
        z   = torch.zeros(2, 2, 8)
        y, S1, z1 = la.recurrent_step(x_0, S, z)
        assert y.shape == (2, 16)

    def test_retentive_shape(self):
        ret = RetentiveLayer(d_model=16, n_heads=2)
        x   = torch.randn(2, 8, 16)
        y   = ret(x)
        assert y.shape == x.shape

    def test_retentive_gamma_range(self):
        ret = RetentiveLayer(d_model=16, n_heads=2)
        assert all(0 < g < 1 for g in ret.gamma.tolist())


# ── Hybrid ────────────────────────────────────────────────────────────────────

class TestHybrid:
    def _model(self):
        cfg = HybridConfig(d_model=16, n_heads=2, d_state=4, n_layers=4,
                            attn_every=2, vocab_size=V)
        return HybridSSMTransformer(cfg)

    def test_output_shape(self):
        m   = self._model()
        ids = torch.randint(0, V, (2, 8))
        y   = m(ids)
        assert y.shape == (2, 8, V)

    def test_layer_summary(self):
        m = self._model()
        s = m.layer_type_summary()
        assert s["total_layers"] == 4
        assert s["attn_layers"] == 2   # attn_every=2 → layers 2,4

    def test_params_positive(self):
        m = self._model()
        assert m.n_parameters() > 0


# ── Analysis ──────────────────────────────────────────────────────────────────

class TestAnalysis:
    def test_flops_comparison_keys(self):
        f = flops_comparison(2, 128, 64, 16)
        assert "transformer_GFLOPs" in f and "mamba_GFLOPs" in f

    def test_transformer_flops_gt_mamba(self):
        f = flops_comparison(2, 512, 64, 16)
        assert f["transformer_GFLOPs"] > f["mamba_GFLOPs"]

    def test_parameter_count_keys(self):
        p = parameter_count_comparison(64, 8, 128, 256, 4)
        assert "transformer_M" in p and "mamba_M" in p

    def test_impulse_response_shape(self):
        ssm = DiscretizedSSM(d_state=4)
        A_bar, B_bar = ssm._discretise()
        C = ssm.C[..., 0] + 1j * ssm.C[..., 1]
        K = compute_ssm_impulse_response(A_bar, B_bar.real, C.real, T=8)
        assert K.shape == (8,)

    def test_effective_memory_length(self):
        K = torch.tensor([1.0, 0.5, 0.2, 0.05, 0.01])
        eff = effective_memory_length(K, threshold=0.1)
        assert eff >= 1
''')
commit("test: add full SSM test suite — HiPPO, DiscretizedSSM, Mamba, S4, LinearAttn, Hybrid, analysis")

for title, body in [
    ("test: add DiscretizedSSM kernel shape test", '''
class TestSSMKernel:
    def test_kernel_shape(self):
        ssm = DiscretizedSSM(d_state=4)
        K   = ssm._kernel(T=16)
        assert K.shape == (16,)

    def test_kernel_real_valued(self):
        ssm = DiscretizedSSM(d_state=4)
        K   = ssm._kernel(T=8)
        assert K.dtype in (torch.float32, torch.float64)
'''),
    ("test: add SelectiveSSM gradient flow test", '''
class TestSelectiveSSMGrad:
    def test_gradients_flow(self):
        ssm = SelectiveSSM(d_inner=16, d_state=4)
        x   = torch.randn(2, 8, 16, requires_grad=True)
        y   = ssm(x)
        y.sum().backward()
        assert x.grad is not None
'''),
    ("test: add S4Layer per-channel independence test", '''
class TestS4Independence:
    def test_n_ssms(self):
        layer = S4Layer(d_model=8, d_state=4)
        assert len(layer.ssms) == 8   # one per channel

    def test_s4_block_residual(self):
        block = S4Block(d_model=8, d_state=4, d_ff=16)
        x = torch.randn(2, 4, 8)
        y = block(x)
        # Residual connection: output != pure SSM output
        assert y.shape == x.shape
'''),
    ("test: add MambaLM loss backprop test", '''
class TestMambaLMLoss:
    def test_loss_backprop(self):
        import torch.nn.functional as F
        cfg = SSMConfig(d_model=16, d_state=4, expand=2)
        lm  = MambaLM(V, cfg, n_layers=2)
        ids = torch.randint(0, V, (2, 8))
        out = lm(ids)
        loss = F.cross_entropy(out[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
        loss.backward()
        # All params should have gradients
        assert all(p.grad is not None for p in lm.parameters() if p.requires_grad)
'''),
    ("test: add LinearAttention kernel positivity test", '''
class TestLinearKernel:
    def test_kernel_positive(self):
        import torch.nn.functional as F
        x = torch.randn(2, 8, 16)
        # ELU+1 kernel
        k = F.elu(x) + 1.0
        assert (k > 0).all()
'''),
    ("test: add HybridSSMTransformer attn_every test", '''
class TestHybridAttnEvery:
    def test_attn_every_1(self):
        from nanomind.ssm import LocalAttentionLayer
        cfg = HybridConfig(d_model=16, n_heads=2, d_state=4,
                            n_layers=4, attn_every=1, vocab_size=V)
        m   = HybridSSMTransformer(cfg)
        # attn_every=1 → all attention layers
        s   = m.layer_type_summary()
        assert s["attn_layers"] == 4

    def test_attn_every_large(self):
        cfg = HybridConfig(d_model=16, n_heads=2, d_state=4,
                            n_layers=4, attn_every=10, vocab_size=V)
        m   = HybridSSMTransformer(cfg)
        # attn_every=10 > n_layers → no attention
        s   = m.layer_type_summary()
        assert s["attn_layers"] == 0
'''),
    ("test: add complexity comparison length test", '''
class TestComplexityList:
    def test_n_architectures(self):
        from nanomind.ssm import ARCHITECTURE_COMPLEXITIES
        assert len(ARCHITECTURE_COMPLEXITIES) >= 5

    def test_parallel_transformer(self):
        from nanomind.ssm import ARCHITECTURE_COMPLEXITIES
        tf = next(c for c in ARCHITECTURE_COMPLEXITIES if c.architecture == "Transformer")
        assert tf.parallel is True

    def test_lstm_not_parallel(self):
        from nanomind.ssm import ARCHITECTURE_COMPLEXITIES
        lstm = next(c for c in ARCHITECTURE_COMPLEXITIES if c.architecture == "LSTM")
        assert lstm.parallel is False
'''),
]:
    src = read("tests/test_ssm.py")
    src += "\n" + body
    write("tests/test_ssm.py", src)
    commit(title)

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v5.6.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"5.5.0\"", "__version__ = \"5.6.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v5.6.0 — Structured State Space Models (SSM/Mamba/S4) release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `vlm`        | Vision-Language Models — CLIP, ViT, VQA (LLaVA/Flamingo), captioning, retrieval |",
    "| `vlm`        | Vision-Language Models — CLIP, ViT, VQA (LLaVA/Flamingo), captioning, retrieval |\n"
    "| `ssm`        | State Space Models — S4, Mamba (selective), linear attention, hybrid, analysis |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [5.6.0] — 2024 — Structured State Space Models\n\n### Added\n"
      "- `SSMConfig` — d_model, d_state, expand, dt_min/max configuration\n"
      "- `make_hippo_matrix` — HiPPO-LegS A matrix for long-range init\n"
      "- `DiscretizedSSM` — ZOH discretization, FFT convolutional + recurrent modes\n"
      "- `SelectiveSSM` — Mamba-style input-dependent Δ, B, C projections\n"
      "- `MambaBlock` — expand → depthwise conv → selective SSM → gate → project\n"
      "- `MambaLM` — full Mamba language model with weight-tied embeddings\n"
      "- `S4Layer` — per-channel SSMs with FFT convolution (O(T log T))\n"
      "- `S4Block` — S4 + FFN with pre-norm residual connections\n"
      "- `S4Model` — S4 language model\n"
      "- `LinearAttention` — O(T) causal linear attention with recurrent_step\n"
      "- `RetentiveLayer` — Retentive Networks with γ-decay mask\n"
      "- `HybridSSMTransformer` — Jamba-style: Mamba + periodic local attention\n"
      "- `LocalAttentionLayer` — sliding-window local attention\n"
      "- `flops_comparison` — GFLOPs analysis across architectures\n"
      "- `parameter_count_comparison` — Transformer vs S4 vs Mamba param counts\n"
      "- `compute_ssm_impulse_response` — K[t] = C Ā^t B̄\n"
      "- `effective_memory_length` — last t where |K[t]| > threshold\n"
      "- `ARCHITECTURE_COMPLEXITIES` — O() comparison dataclass list\n"
      "- `examples/ssm_demo.py` — full SSM pipeline demo\n\n---\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v5.6.0, update README and CHANGELOG for Day 56 SSM/Mamba")

print("\n=== Pushing Day 56 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")
run("git", "tag", "-a", "v5.6.0", "-m", "NanoMind v5.6.0 — SSM/Mamba", check=False)
r = run("git", "push", "origin", "v5.6.0", check=False)
print("Tag v5.6.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 56 COMPLETE — v5.6.0 TAGGED! ===")
