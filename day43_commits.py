"""
day43_commits.py — 20 atomic commits for Day 43: Diffusion Language Models.
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

print("\n=== DAY 43: Diffusion Language Models — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — diffusion package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/diffusion/__init__.py",
      '"""NanoMind Diffusion sub-package — Diffusion Language Models."""\n')
commit("feat: add nanomind/diffusion/ package skeleton for Diffusion Language Models")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — noise schedules
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/diffusion/schedule.py", '''\
"""
nanomind/diffusion/schedule.py — Noise schedules for diffusion models.

## Diffusion Model Overview

Diffusion models define two processes:

Forward (noising):
  q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) x_{t-1}, β_t I)
  Gradually adds Gaussian noise over T steps.
  At t=T: x_T ≈ N(0, I)

Reverse (denoising):
  p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
  A neural network learns to remove noise step by step.

Key identity (reparametrisation):
  x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε,  ε ~ N(0, I)
  where ᾱ_t = Π_{s=1}^t (1 - β_s)

This allows sampling x_t at any timestep in one step,
making training efficient (random t per sample).

## Noise Schedules

Linear (Ho et al., 2020):
  β_t = β_start + t/T × (β_end - β_start)

Cosine (Nichol & Dhariwal, 2021):
  ᾱ_t = cos²( (t/T + s) / (1+s) × π/2 )
  Avoids too much noise at early steps.

Sigmoid:
  β_t = sigmoid( (t - T/2) / T × 6 )
  Smooth transition.

References:
  Ho et al. (2020) "Denoising Diffusion Probabilistic Models" DDPM
  https://arxiv.org/abs/2006.11239

  Nichol & Dhariwal (2021) "Improved DDPM"
  https://arxiv.org/abs/2102.09672

  Song et al. (2020) "Score-Based Generative Modeling" DDIM
  https://arxiv.org/abs/2010.02502
"""

from __future__ import annotations
import math
import torch


class NoiseSchedule:
    """
    Precomputed noise schedule for diffusion models.

    Provides β_t, ᾱ_t, and related quantities for all timesteps.

    Args:
        n_steps:    Total diffusion timesteps T.
        schedule:   ``"linear"``, ``"cosine"``, or ``"sigmoid"``.
        beta_start: Starting β (linear only).
        beta_end:   Ending β (linear only).

    Example::

        sched = NoiseSchedule(n_steps=1000, schedule="cosine")
        # Sample noisy x at timestep t:
        x_noisy, noise = sched.q_sample(x0, t=500)
        # Get SNR at t:
        snr_t = sched.snr(500)
    """

    def __init__(
        self,
        n_steps:    int   = 1000,
        schedule:   str   = "cosine",
        beta_start: float = 1e-4,
        beta_end:   float = 0.02,
    ) -> None:
        assert schedule in ("linear", "cosine", "sigmoid")
        self.n_steps  = n_steps
        self.schedule = schedule

        betas = self._make_betas(n_steps, schedule, beta_start, beta_end)
        self.register(betas)

    def _make_betas(
        self,
        T:     int,
        sched: str,
        b0:    float,
        b1:    float,
    ) -> torch.Tensor:
        t = torch.linspace(0, T - 1, T)
        if sched == "linear":
            return torch.linspace(b0, b1, T)
        elif sched == "cosine":
            s   = 0.008
            ft  = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
            f0  = math.cos((s / (1 + s)) * math.pi / 2) ** 2
            betas = 1 - ft / f0
            return betas.clamp(0.0001, 0.9999)
        elif sched == "sigmoid":
            x     = 6 * (t / T - 0.5)
            betas = torch.sigmoid(x)
            # Normalise to [b0, b1]
            betas = b0 + (b1 - b0) * (betas - betas.min()) / (betas.max() - betas.min() + 1e-8)
            return betas

    def register(self, betas: torch.Tensor) -> None:
        """Precompute all schedule quantities."""
        self.betas      = betas
        alphas          = 1.0 - betas
        self.alphas     = alphas
        self.alpha_bars = torch.cumprod(alphas, dim=0)
        # Shifted: ᾱ_{t-1}
        prev_ab         = torch.cat([torch.tensor([1.0]), self.alpha_bars[:-1]])
        self.alpha_bars_prev = prev_ab
        # For reverse process
        self.sqrt_alpha_bars      = self.alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars = (1 - self.alpha_bars).sqrt()
        self.log_one_minus_alpha_bars  = (1 - self.alpha_bars).log()
        self.sqrt_recip_alphas    = (1 / alphas).sqrt()
        # Posterior variance for DDPM
        self.posterior_variance   = betas * (1 - prev_ab) / (1 - self.alpha_bars + 1e-8)
        self.posterior_log_var    = self.posterior_variance.clamp(min=1e-20).log()

    def q_sample(
        self,
        x0: torch.Tensor,
        t:  torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: sample x_t from x_0.

        x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε

        Args:
            x0:    ``(B, ...)`` clean input.
            t:     ``(B,)`` timestep indices.
            noise: Optional pre-sampled noise.

        Returns:
            ``(x_t, noise)``
        """
        if noise is None:
            noise = torch.randn_like(x0)
        s_ab = self.sqrt_alpha_bars[t]
        s_1m = self.sqrt_one_minus_alpha_bars[t]
        # Broadcast to x0 shape
        while s_ab.dim() < x0.dim():
            s_ab = s_ab.unsqueeze(-1)
            s_1m = s_1m.unsqueeze(-1)
        return s_ab * x0 + s_1m * noise, noise

    def snr(self, t: int) -> float:
        """Signal-to-noise ratio at timestep t."""
        ab = self.alpha_bars[t].item()
        return ab / (1 - ab + 1e-8)

    def predict_x0(
        self,
        x_t: torch.Tensor,
        t:   torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise ε.

        x_0 = (x_t - sqrt(1-ᾱ_t) ε) / sqrt(ᾱ_t)
        """
        s_ab = self.sqrt_alpha_bars[t]
        s_1m = self.sqrt_one_minus_alpha_bars[t]
        while s_ab.dim() < x_t.dim():
            s_ab = s_ab.unsqueeze(-1)
            s_1m = s_1m.unsqueeze(-1)
        return (x_t - s_1m * eps) / (s_ab + 1e-8)

    def to_dict(self) -> dict:
        return {
            "n_steps":   self.n_steps,
            "schedule":  self.schedule,
            "beta_min":  round(self.betas.min().item(), 6),
            "beta_max":  round(self.betas.max().item(), 6),
            "snr_t0":    round(self.snr(0), 2),
            "snr_tT":    round(self.snr(self.n_steps - 1), 6),
        }
''')
commit("feat: add NoiseSchedule — linear/cosine/sigmoid betas, q_sample, snr, predict_x0, alpha_bars")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — denoising network (transformer-based)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/diffusion/denoiser.py", '''\
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
''')
commit("feat: add SinusoidalTimeEmbedding, DiffusionTransformerBlock (AdaLN), DiffusionDenoiser (eps prediction)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — DDPM sampler
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/diffusion/ddpm.py", '''\
"""
nanomind/diffusion/ddpm.py — DDPM sampler (Denoising Diffusion Probabilistic Models).

DDPM (Ho et al., 2020) training:
  1. Sample x_0 from data
  2. Sample t ~ Uniform(1, T)
  3. Sample ε ~ N(0, I)
  4. Compute x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
  5. Train: minimize ||ε_θ(x_t, t) - ε||²

DDPM sampling (reverse process, T steps):
  1. Start from x_T ~ N(0, I)
  2. For t = T, T-1, ..., 1:
       μ_t = (1/sqrt(α_t)) × (x_t - β_t/sqrt(1-ᾱ_t) × ε_θ(x_t, t))
       x_{t-1} = μ_t + sqrt(β_t) × z,  z ~ N(0, I) if t > 1 else 0

This requires T=1000 model evaluations per sample — slow!
DDIM accelerates this to 50-100 steps.

Reference:
  Ho et al. (2020) https://arxiv.org/abs/2006.11239
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

from nanomind.diffusion.schedule import NoiseSchedule
from nanomind.diffusion.denoiser import DiffusionDenoiser
from nanomind.utils.logger import get_logger

log = get_logger("diffusion.ddpm")


class DDPMTrainer:
    """
    DDPM training: simple noise prediction loss.

    Args:
        denoiser: :class:`DiffusionDenoiser`.
        schedule: :class:`NoiseSchedule`.

    Example::

        trainer = DDPMTrainer(denoiser, schedule)
        loss    = trainer.loss(token_ids)
        loss.backward()
    """

    def __init__(
        self,
        denoiser: DiffusionDenoiser,
        schedule: NoiseSchedule,
    ) -> None:
        self.denoiser = denoiser
        self.schedule = schedule

    def loss(
        self,
        token_ids: torch.Tensor,
        loss_type: str = "simple",
    ) -> torch.Tensor:
        """
        Compute DDPM training loss.

        Args:
            token_ids: ``(B, T)`` input token IDs.
            loss_type: ``"simple"`` (ε prediction) or ``"vlb"`` (variational lower bound).

        Returns:
            Scalar loss tensor.
        """
        B  = token_ids.shape[0]
        x0 = self.denoiser.embed_tokens(token_ids)          # (B, T, D)
        t  = torch.randint(0, self.schedule.n_steps, (B,))  # random timestep

        x_t, noise = self.schedule.q_sample(x0, t)          # (B, T, D)
        eps_pred   = self.denoiser(x_t, t)                   # (B, T, D)

        if loss_type == "simple":
            return F.mse_loss(eps_pred, noise)
        elif loss_type == "vlb":
            # SNR-weighted loss (Min-SNR, Hang et al., 2023)
            snr    = self.schedule.alpha_bars[t] / (1 - self.schedule.alpha_bars[t] + 1e-8)
            weight = (snr / (snr + 1)).unsqueeze(-1).unsqueeze(-1)
            return (weight * F.mse_loss(eps_pred, noise, reduction="none")).mean()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type!r}")

    def loss_at_t(self, token_ids: torch.Tensor, t: int) -> float:
        """Compute loss at a specific timestep (for analysis)."""
        B  = token_ids.shape[0]
        x0 = self.denoiser.embed_tokens(token_ids)
        ts = torch.full((B,), t, dtype=torch.long)
        x_t, noise = self.schedule.q_sample(x0, ts)
        eps_pred   = self.denoiser(x_t, ts)
        return F.mse_loss(eps_pred, noise).item()


class DDPMSampler:
    """
    DDPM ancestral sampler: full T-step reverse process.

    Args:
        denoiser: Trained :class:`DiffusionDenoiser`.
        schedule: :class:`NoiseSchedule`.

    Example::

        sampler  = DDPMSampler(denoiser, schedule)
        token_ids = sampler.sample(batch_size=2, seq_len=16, n_steps=100)
    """

    def __init__(
        self,
        denoiser: DiffusionDenoiser,
        schedule: NoiseSchedule,
    ) -> None:
        self.denoiser = denoiser
        self.schedule = schedule

    @torch.no_grad()
    def p_sample_step(
        self,
        x_t: torch.Tensor,
        t:   int,
    ) -> torch.Tensor:
        """
        Single DDPM reverse step: x_t → x_{t-1}.

        Args:
            x_t: ``(B, T, D)`` noisy embeddings at step t.
            t:   Current timestep.

        Returns:
            ``(B, T, D)`` less noisy embeddings x_{t-1}.
        """
        B    = x_t.shape[0]
        ts   = torch.full((B,), t, dtype=torch.long)
        eps  = self.denoiser(x_t, ts)

        alpha    = self.schedule.alphas[t]
        alpha_b  = self.schedule.alpha_bars[t]
        beta     = self.schedule.betas[t]
        s_recip  = self.schedule.sqrt_recip_alphas[t]
        s_1m     = self.schedule.sqrt_one_minus_alpha_bars[t]

        mu = s_recip * (x_t - beta / s_1m * eps)

        if t > 0:
            noise   = torch.randn_like(x_t)
            post_var = self.schedule.posterior_variance[t]
            x_prev  = mu + post_var.sqrt() * noise
        else:
            x_prev  = mu
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len:    int = 16,
        n_steps:    int | None = None,
    ) -> torch.Tensor:
        """
        Full DDPM sampling.

        Args:
            batch_size: Number of sequences to generate.
            seq_len:    Sequence length.
            n_steps:    Steps to use (default: schedule.n_steps).

        Returns:
            ``(B, T)`` token IDs.
        """
        D     = self.denoiser.d_model
        T     = self.schedule.n_steps if n_steps is None else n_steps
        x     = torch.randn(batch_size, seq_len, D)   # x_T ~ N(0,I)

        steps = list(range(T - 1, -1, -1))[:T]
        for t in steps:
            x = self.p_sample_step(x, t)

        return self.denoiser.decode_to_tokens(x)

    @property
    def schedule_summary(self) -> dict:
        return self.schedule.to_dict()
''')
commit("feat: add DDPMTrainer (simple/vlb loss, loss_at_t), DDPMSampler (p_sample_step, sample)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — DDIM sampler (fast)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/diffusion/ddim.py", '''\
"""
nanomind/diffusion/ddim.py — DDIM sampler (Denoising Diffusion Implicit Models).

DDIM (Song et al., 2020) is a non-Markovian reverse process that:
  1. Uses the same trained DDPM model (no retraining needed)
  2. Samples in S << T steps (e.g., 50 vs 1000)
  3. Produces deterministic or stochastic samples

DDIM update rule:
  x_{t-1} = sqrt(ᾱ_{t-1}) × x̂_0(x_t, t)
           + sqrt(1-ᾱ_{t-1} - η²σ²_t) × ε_θ(x_t, t)
           + η × σ_t × z

where:
  x̂_0 = (x_t - sqrt(1-ᾱ_t) ε_θ) / sqrt(ᾱ_t)    (predicted x_0)
  σ_t  = sqrt((1-ᾱ_{t-1})/(1-ᾱ_t) × β_t)
  η=0: deterministic (DDIM), η=1: stochastic (DDPM)

Speed: 20× faster than DDPM with comparable quality.

Reference:
  Song et al. (2020) "Denoising Diffusion Implicit Models"
  https://arxiv.org/abs/2010.02502
"""

from __future__ import annotations
import torch
import numpy as np

from nanomind.diffusion.schedule import NoiseSchedule
from nanomind.diffusion.denoiser import DiffusionDenoiser


class DDIMSampler:
    """
    DDIM fast sampler — same model, fewer steps.

    Args:
        denoiser: Trained :class:`DiffusionDenoiser`.
        schedule: :class:`NoiseSchedule`.
        eta:      Stochasticity (0 = deterministic DDIM, 1 = DDPM).

    Example::

        sampler = DDIMSampler(denoiser, schedule, eta=0.0)
        tokens  = sampler.sample(batch_size=2, seq_len=16, n_steps=50)
    """

    def __init__(
        self,
        denoiser: DiffusionDenoiser,
        schedule: NoiseSchedule,
        eta:      float = 0.0,
    ) -> None:
        self.denoiser = denoiser
        self.schedule = schedule
        self.eta      = eta

    def _make_timesteps(self, n_steps: int) -> list[int]:
        """Create evenly-spaced timestep sequence for DDIM."""
        T    = self.schedule.n_steps
        step = T // n_steps
        ts   = list(range(0, T, step))[::-1]
        return ts

    @torch.no_grad()
    def ddim_step(
        self,
        x_t:   torch.Tensor,
        t:     int,
        t_prev: int,
    ) -> torch.Tensor:
        """
        Single DDIM reverse step from timestep t to t_prev.

        Args:
            x_t:    ``(B, T, D)`` noisy embeddings at step t.
            t:      Current step.
            t_prev: Previous step (< t).

        Returns:
            ``(B, T, D)`` x at step t_prev.
        """
        B    = x_t.shape[0]
        ts   = torch.full((B,), t, dtype=torch.long)
        eps  = self.denoiser(x_t, ts)

        ab_t    = self.schedule.alpha_bars[t]
        ab_prev = self.schedule.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predict x_0
        x0_pred = (x_t - (1 - ab_t).sqrt() * eps) / (ab_t.sqrt() + 1e-8)

        # DDIM sigma
        sigma = self.eta * ((1 - ab_prev) / (1 - ab_t + 1e-8)
                             * (1 - ab_t / (ab_prev + 1e-8))).clamp(0).sqrt()

        # Direction toward x_t
        dir_xt = (1 - ab_prev - sigma ** 2).clamp(0).sqrt() * eps

        # Noise term
        noise  = torch.randn_like(x_t) if self.eta > 0 else torch.zeros_like(x_t)

        x_prev = ab_prev.sqrt() * x0_pred + dir_xt + sigma * noise
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len:    int = 16,
        n_steps:    int = 50,
    ) -> torch.Tensor:
        """
        Fast DDIM sampling.

        Args:
            batch_size: Number of sequences.
            seq_len:    Sequence length.
            n_steps:    Number of denoising steps (50 by default).

        Returns:
            ``(B, T)`` token IDs.
        """
        D     = self.denoiser.d_model
        x     = torch.randn(batch_size, seq_len, D)   # start from noise
        ts    = self._make_timesteps(n_steps)

        for i, t in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            x      = self.ddim_step(x, t, t_prev)

        return self.denoiser.decode_to_tokens(x)

    @torch.no_grad()
    def encode(
        self,
        token_ids: torch.Tensor,
        n_steps:   int = 50,
    ) -> torch.Tensor:
        """
        DDIM inversion: encode clean tokens to noise (for editing).

        Args:
            token_ids: ``(B, T)`` clean token IDs.
            n_steps:   Inversion steps.

        Returns:
            ``(B, T, D)`` noise tensor x_T.
        """
        x    = self.denoiser.embed_tokens(token_ids).float()
        ts   = self._make_timesteps(n_steps)[::-1]   # reversed order

        for i, t in enumerate(ts):
            t_next = ts[i + 1] if i + 1 < len(ts) else self.schedule.n_steps - 1
            B      = x.shape[0]
            ts_b   = torch.full((B,), t, dtype=torch.long)
            eps    = self.denoiser(x, ts_b)
            ab     = self.schedule.alpha_bars[t]
            ab_n   = self.schedule.alpha_bars[t_next]
            x      = ab_n.sqrt() * (x - (1 - ab).sqrt() * eps) / (ab.sqrt() + 1e-8) \
                     + (1 - ab_n).sqrt() * eps
        return x
''')
commit("feat: add DDIMSampler — n_steps fast sampling, ddim_step, encode (DDIM inversion), eta control")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — masked diffusion (D3PM-style)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/diffusion/masked.py", '''\
"""
nanomind/diffusion/masked.py — Masked Diffusion Language Model (D3PM / MDM).

Masked diffusion treats the forward process as progressive masking:
  q(x_t | x_{t-1}): independently mask each token with probability β_t
  q(x_T | x_0):     all tokens are masked with high probability

The model learns to recover masked tokens:
  p_θ(x_0 | x_t) ≈ BERT-style MLM objective

This is closely related to BERT (masked language modelling) but:
  - BERT masks 15% randomly, fixed
  - MDM masks at varying rates controlled by the schedule
  - MDM has a principled generative process

At generation time:
  1. Start with all-MASK tokens
  2. Iteratively unmask tokens using learned p_θ(x_0 | x_t)

Used in: MDLM (Sahoo et al., 2024), SEDD (Lou et al., 2024).

Reference:
  Austin et al. (2021) "Structured Denoising Diffusion" D3PM
  https://arxiv.org/abs/2107.03006

  Sahoo et al. (2024) MDLM: https://arxiv.org/abs/2406.07524
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


MASK_ID = 1   # Token ID used as [MASK]


class MaskedDiffusionLM(nn.Module):
    """
    Masked Diffusion Language Model.

    Forward: mask tokens at rate controlled by schedule.
    Reverse: predict original tokens from masked input (MLM).

    Args:
        vocab_size: Vocabulary size (including MASK_ID=1).
        d_model:    Model dimension.
        n_layers:   Transformer layers.
        n_heads:    Attention heads.
        max_seq:    Max sequence length.

    Example::

        mdlm = MaskedDiffusionLM(vocab_size=100, d_model=64)
        loss = mdlm.loss(token_ids, t=50, T=100)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int = 64,
        n_layers:   int = 3,
        n_heads:    int = 4,
        max_seq:    int = 64,
    ) -> None:
        super().__init__()
        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.tok         = nn.Embedding(vocab_size, d_model)
        self.pos         = nn.Embedding(max_seq, d_model)
        self.t_emb       = nn.Embedding(1001, d_model)
        self.blocks      = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.head        = nn.Linear(d_model, vocab_size)

    def _mask_rate(self, t: int, T: int) -> float:
        """Linear mask rate: 0 at t=0, 1 at t=T."""
        return t / max(T, 1)

    def forward_mask(
        self,
        x:  torch.Tensor,
        t:  int,
        T:  int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: randomly mask tokens.

        Args:
            x:  ``(B, L)`` clean token IDs.
            t:  Diffusion step.
            T:  Total steps.

        Returns:
            ``(x_masked, mask)`` where mask is True at masked positions.
        """
        rate   = self._mask_rate(t, T)
        mask   = torch.rand_like(x.float()) < rate
        x_mask = x.clone()
        x_mask[mask] = MASK_ID
        return x_mask, mask

    def predict(
        self,
        x_masked: torch.Tensor,
        t:        int,
    ) -> torch.Tensor:
        """
        Predict logits for all positions.

        Args:
            x_masked: ``(B, L)`` masked token IDs.
            t:        Diffusion step.

        Returns:
            ``(B, L, V)`` logits.
        """
        B, L    = x_masked.shape
        pos     = torch.arange(L)
        ts      = torch.full((B,), min(t, 1000), dtype=torch.long)
        h       = self.tok(x_masked) + self.pos(pos) + self.t_emb(ts).unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        return self.head(h)

    def loss(
        self,
        token_ids: torch.Tensor,
        t:         int,
        T:         int = 1000,
    ) -> torch.Tensor:
        """
        Compute masked diffusion training loss.

        Only compute loss on masked positions (like BERT MLM).
        """
        x_masked, mask = self.forward_mask(token_ids, t, T)
        logits         = self.predict(x_masked, t)         # (B, L, V)

        # Only penalise masked tokens
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)

        flat_logits = logits[mask]                          # (M, V)
        flat_labels = token_ids[mask]                       # (M,)
        return F.cross_entropy(flat_logits, flat_labels)

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        seq_len:    int = 16,
        T:          int = 20,
    ) -> torch.Tensor:
        """
        Generate text via iterative unmasking.

        Start fully masked, unmask top-confidence tokens each step.
        """
        x = torch.full((batch_size, seq_len), MASK_ID, dtype=torch.long)

        for step in range(T, 0, -1):
            logits = self.predict(x, step)   # (B, L, V)
            probs  = torch.softmax(logits, dim=-1)
            tokens = probs.argmax(dim=-1)    # (B, L)

            # Only unmask: keep already unmasked, sample new ones
            is_masked     = (x == MASK_ID)
            confidence, _ = probs.max(dim=-1)   # (B, L)
            # Unmask top-confidence masked positions
            n_unmask = max(1, int(seq_len * step / T))
            for b in range(batch_size):
                masked_pos = is_masked[b].nonzero(as_tuple=True)[0]
                if len(masked_pos) == 0:
                    continue
                conf_b = confidence[b][masked_pos]
                n      = min(n_unmask, len(masked_pos))
                top_k  = conf_b.topk(n).indices
                chosen = masked_pos[top_k]
                x[b, chosen] = tokens[b, chosen]
        return x
''')
commit("feat: add MaskedDiffusionLM — D3PM-style forward_mask, predict, loss, iterative generate()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — classifier-free guidance
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/diffusion/guidance.py", '''\
"""
nanomind/diffusion/guidance.py — Classifier-Free Guidance (CFG) for diffusion LMs.

## Classifier-Free Guidance (Ho & Salimans, 2022)

CFG mixes conditional and unconditional noise predictions:
  ε̃ = ε_unconditional + γ × (ε_conditional - ε_unconditional)

where γ is the guidance scale (guidance strength).
Higher γ → stronger conditioning → less diversity, more relevance.

Used in:
  - Stable Diffusion (images)
  - DALL-E 2, Imagen
  - InstructDiffusion (text editing)

For language: condition on a prefix, label, or style embedding.

Training: randomly drop condition with probability p_drop (10-20%).
  During training: p(keep_condition) = 1 - p_drop
  At inference: always use condition

Reference:
  Ho & Salimans (2022) "Classifier-Free Diffusion Guidance"
  https://arxiv.org/abs/2207.12598
"""

from __future__ import annotations
import torch
import torch.nn as nn


class ClassifierFreeGuidance:
    """
    Classifier-Free Guidance wrapper for diffusion denoising.

    Args:
        denoiser:       Conditional denoiser (accepts optional condition).
        guidance_scale: γ (1.0 = no guidance, >1.0 = stronger conditioning).
        p_uncond:       Probability of unconditional training.

    Example::

        cfg     = ClassifierFreeGuidance(denoiser, guidance_scale=7.5)
        eps     = cfg.guided_predict(x_t, t, condition=cond_emb)
    """

    def __init__(
        self,
        denoiser:       nn.Module,
        guidance_scale: float = 7.5,
        p_uncond:       float = 0.1,
    ) -> None:
        self.denoiser       = denoiser
        self.guidance_scale = guidance_scale
        self.p_uncond       = p_uncond
        self._null_cond:    torch.Tensor | None = None

    def set_null_condition(self, condition: torch.Tensor) -> None:
        """Register the null/unconditional embedding (e.g., zero vector)."""
        self._null_cond = torch.zeros_like(condition)

    def train_mask(self, condition: torch.Tensor) -> torch.Tensor:
        """
        During training, randomly drop condition for unconditional learning.

        Args:
            condition: ``(B, D_c)`` condition embeddings.

        Returns:
            Masked condition with p_uncond probability of zeros.
        """
        mask = torch.rand(condition.shape[0]) < self.p_uncond
        null = torch.zeros_like(condition)
        cond = condition.clone()
        cond[mask] = null[mask]
        return cond

    @torch.no_grad()
    def guided_predict(
        self,
        x_t:       torch.Tensor,
        t:         torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classifier-free guided noise prediction.

        ε̃ = ε_uncond + γ × (ε_cond - ε_uncond)

        Args:
            x_t:       ``(B, T, D)`` noisy embeddings.
            t:         ``(B,)`` timesteps.
            condition: ``(B, D_c)`` condition embedding.

        Returns:
            ``(B, T, D)`` guided noise prediction.
        """
        # Concatenate conditional and unconditional batches
        null_cond = torch.zeros_like(condition)
        x_cat     = torch.cat([x_t, x_t], dim=0)
        t_cat     = torch.cat([t, t], dim=0)
        cond_cat  = torch.cat([condition, null_cond], dim=0)

        # Single forward pass for both
        eps_both  = self.denoiser(x_cat, t_cat)
        eps_cond, eps_uncond = eps_both.chunk(2, dim=0)

        # Guidance interpolation
        return eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)

    def guidance_scale_schedule(
        self,
        t: int,
        T: int,
        min_scale: float = 1.0,
        max_scale: float | None = None,
    ) -> float:
        """
        Dynamic guidance scale: increase as t decreases (fine detail phase).

        Args:
            t:         Current timestep.
            T:         Total timesteps.
            min_scale: Minimum guidance scale.
            max_scale: Maximum guidance scale.
        """
        max_s = max_scale or self.guidance_scale
        # Linear: low guidance at high noise, high guidance at low noise
        frac  = 1.0 - t / max(T, 1)
        return min_scale + frac * (max_s - min_scale)
''')
commit("feat: add ClassifierFreeGuidance — guided_predict (CFG interpolation), train_mask, dynamic scale schedule")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — diffusion LM pipeline
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/diffusion/pipeline.py", '''\
"""
nanomind/diffusion/pipeline.py — End-to-end Diffusion LM pipeline.

Unified interface combining:
  - NoiseSchedule
  - DiffusionDenoiser
  - DDPMTrainer / DDIMSampler
  - Masked diffusion option
"""

from __future__ import annotations
import torch
import torch.optim as optim

from nanomind.diffusion.schedule import NoiseSchedule
from nanomind.diffusion.denoiser import DiffusionDenoiser
from nanomind.diffusion.ddpm import DDPMTrainer, DDPMSampler
from nanomind.diffusion.ddim import DDIMSampler
from nanomind.diffusion.masked import MaskedDiffusionLM
from nanomind.utils.logger import get_logger

log = get_logger("diffusion.pipeline")


class DiffusionLMPipeline:
    """
    End-to-end Diffusion Language Model pipeline.

    Supports two modes:
      - ``"continuous"``: Gaussian diffusion in embedding space (DDPM/DDIM)
      - ``"masked"``:     Token masking diffusion (D3PM/MDLM)

    Args:
        vocab_size: Vocabulary size.
        d_model:    Embedding dimension.
        n_layers:   Transformer layers.
        n_heads:    Attention heads.
        max_seq:    Maximum sequence length.
        n_steps:    Diffusion steps.
        schedule:   Noise schedule type.
        mode:       ``"continuous"`` or ``"masked"``.

    Example::

        pipe = DiffusionLMPipeline(vocab_size=100, d_model=64, mode="continuous")
        pipe.train_step(token_ids)
        tokens = pipe.generate(batch_size=2, seq_len=16, n_steps=20)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int   = 64,
        n_layers:   int   = 3,
        n_heads:    int   = 4,
        max_seq:    int   = 32,
        n_steps:    int   = 100,
        schedule:   str   = "cosine",
        mode:       str   = "continuous",
        lr:         float = 1e-3,
    ) -> None:
        assert mode in ("continuous", "masked")
        self.mode    = mode
        self.n_steps = n_steps

        if mode == "continuous":
            self.noise_schedule = NoiseSchedule(n_steps=n_steps, schedule=schedule)
            self.denoiser       = DiffusionDenoiser(
                vocab_size, d_model, n_layers, n_heads, max_seq
            )
            self._trainer = DDPMTrainer(self.denoiser, self.noise_schedule)
            self._ddpm    = DDPMSampler(self.denoiser, self.noise_schedule)
            self._ddim    = DDIMSampler(self.denoiser, self.noise_schedule, eta=0.0)
            self.opt      = optim.Adam(self.denoiser.parameters(), lr=lr)
        else:
            self.denoiser = MaskedDiffusionLM(
                vocab_size, d_model, n_layers, n_heads, max_seq
            )
            self.opt      = optim.Adam(self.denoiser.parameters(), lr=lr)

    def train_step(
        self,
        token_ids: torch.Tensor,
        loss_type: str = "simple",
    ) -> float:
        """
        Single training step.

        Args:
            token_ids: ``(B, T)`` token IDs.
            loss_type: ``"simple"`` or ``"vlb"`` (continuous mode only).

        Returns:
            Loss value.
        """
        self.opt.zero_grad()
        if self.mode == "continuous":
            loss = self._trainer.loss(token_ids, loss_type=loss_type)
        else:
            t    = torch.randint(1, self.n_steps + 1, (1,)).item()
            loss = self.denoiser.loss(token_ids, t=t, T=self.n_steps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        seq_len:    int = 16,
        n_steps:    int = 20,
        sampler:    str = "ddim",
    ) -> torch.Tensor:
        """
        Generate token sequences.

        Args:
            batch_size: Number of sequences.
            seq_len:    Sequence length.
            n_steps:    Sampling steps.
            sampler:    ``"ddpm"``, ``"ddim"``, or ``"masked"`` (auto).

        Returns:
            ``(B, T)`` token IDs.
        """
        if self.mode == "masked":
            return self.denoiser.generate(batch_size, seq_len, T=n_steps)
        elif sampler == "ddim":
            return self._ddim.sample(batch_size, seq_len, n_steps=n_steps)
        else:
            return self._ddpm.sample(batch_size, seq_len, n_steps=n_steps)

    def info(self) -> dict:
        """Return pipeline configuration summary."""
        n_params = sum(p.numel() for p in self.denoiser.parameters())
        d = {
            "mode":     self.mode,
            "n_steps":  self.n_steps,
            "n_params": n_params,
        }
        if self.mode == "continuous":
            d["schedule"] = self.noise_schedule.to_dict()
        return d
''')
commit("feat: add DiffusionLMPipeline — continuous/masked modes, train_step, generate (DDPM/DDIM/masked)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — diffusion __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/diffusion/__init__.py", '''\
"""NanoMind Diffusion sub-package — Diffusion Language Models.

Implements the full diffusion language model stack:
  1. NoiseSchedule      — linear/cosine/sigmoid β schedules, q_sample
  2. DiffusionDenoiser  — AdaLN transformer, sinusoidal time emb, eps prediction
  3. DDPMTrainer        — simple/vlb loss, loss_at_t
  4. DDPMSampler        — ancestral sampling, p_sample_step
  5. DDIMSampler        — fast sampling (50 steps), DDIM inversion
  6. MaskedDiffusionLM  — D3PM-style masked diffusion, iterative generate
  7. ClassifierFreeGuidance — CFG interpolation, guided_predict
  8. DiffusionLMPipeline    — unified continuous/masked pipeline

Primary exports:
    - :class:`NoiseSchedule`         — q_sample, snr, predict_x0, alpha_bars
    - :class:`DiffusionDenoiser`     — forward (eps pred), embed_tokens, decode_to_tokens
    - :class:`SinusoidalTimeEmbedding` — sinusoidal timestep encoding
    - :class:`DDPMTrainer`           — loss (simple/vlb), loss_at_t
    - :class:`DDPMSampler`           — sample, p_sample_step
    - :class:`DDIMSampler`           — sample (fast), ddim_step, encode
    - :class:`MaskedDiffusionLM`     — forward_mask, predict, loss, generate
    - :class:`ClassifierFreeGuidance` — guided_predict, train_mask, guidance_scale_schedule
    - :class:`DiffusionLMPipeline`   — train_step, generate, info
"""

from nanomind.diffusion.schedule import NoiseSchedule
from nanomind.diffusion.denoiser import (
    DiffusionDenoiser, SinusoidalTimeEmbedding, DiffusionTransformerBlock
)
from nanomind.diffusion.ddpm import DDPMTrainer, DDPMSampler
from nanomind.diffusion.ddim import DDIMSampler
from nanomind.diffusion.masked import MaskedDiffusionLM, MASK_ID
from nanomind.diffusion.guidance import ClassifierFreeGuidance
from nanomind.diffusion.pipeline import DiffusionLMPipeline

__all__ = [
    "NoiseSchedule",
    "DiffusionDenoiser", "SinusoidalTimeEmbedding", "DiffusionTransformerBlock",
    "DDPMTrainer", "DDPMSampler",
    "DDIMSampler",
    "MaskedDiffusionLM", "MASK_ID",
    "ClassifierFreeGuidance",
    "DiffusionLMPipeline",
]
''')
commit("refactor: export all diffusion components from nanomind/diffusion/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — example
# ══════════════════════════════════════════════════════════════════════════════
write("examples/diffusion_demo.py", '''\
"""
examples/diffusion_demo.py — NanoMind Diffusion Language Model demo.

Demonstrates:
  1. NoiseSchedule: linear, cosine, sigmoid — q_sample, SNR
  2. DiffusionDenoiser: AdaLN transformer, sinusoidal time embedding
  3. DDPM: training loss, ancestral sampling
  4. DDIM: 10× faster sampling
  5. MaskedDiffusionLM: D3PM-style token masking
  6. ClassifierFreeGuidance: CFG noise prediction
  7. DiffusionLMPipeline: end-to-end training + generation

Usage:
    python examples/diffusion_demo.py
"""
import torch
from nanomind.diffusion import (
    NoiseSchedule, DiffusionDenoiser, SinusoidalTimeEmbedding,
    DDPMTrainer, DDPMSampler, DDIMSampler,
    MaskedDiffusionLM, ClassifierFreeGuidance,
    DiffusionLMPipeline,
)

V   = 32
print("=" * 60)
print("NanoMind Diffusion Language Model Demo")
print("=" * 60)

# ── NoiseSchedule ─────────────────────────────────────────────────────────────
print("\n── Noise Schedules ──")
for sched in ["linear", "cosine", "sigmoid"]:
    ns = NoiseSchedule(n_steps=100, schedule=sched)
    print(f"  {sched:8}: {ns.to_dict()}")

ns = NoiseSchedule(n_steps=100, schedule="cosine")
x0 = torch.randn(2, 8, 16)     # (B, T, D) continuous embeddings
t  = torch.tensor([10, 50])
x_t, noise = ns.q_sample(x0, t)
print(f"\n  q_sample: x_t.shape={tuple(x_t.shape)}, noise.shape={tuple(noise.shape)}")
print(f"  SNR at t=0: {ns.snr(0):.2f}, t=50: {ns.snr(50):.4f}, t=99: {ns.snr(99):.6f}")

# ── SinusoidalTimeEmbedding ───────────────────────────────────────────────────
print("\n── Sinusoidal Time Embedding ──")
te = SinusoidalTimeEmbedding(dim=32)
t_emb = te(torch.tensor([0, 50, 99]))
print(f"  t_emb shape: {tuple(t_emb.shape)}")

# ── DiffusionDenoiser ─────────────────────────────────────────────────────────
print("\n── DiffusionDenoiser (AdaLN Transformer) ──")
denoiser = DiffusionDenoiser(vocab_size=V, d_model=32, n_layers=2, n_heads=2, max_seq=16)
print(f"  Params: {denoiser.n_params:,}")
x_noisy = torch.randn(2, 8, 32)
t_ids   = torch.randint(0, 100, (2,))
with torch.no_grad():
    eps_pred = denoiser(x_noisy, t_ids)
print(f"  eps_pred shape: {tuple(eps_pred.shape)}")
tok_ids = denoiser.decode_to_tokens(x_noisy)
print(f"  decode_to_tokens: {tuple(tok_ids.shape)}")

# ── DDPM Training ─────────────────────────────────────────────────────────────
print("\n── DDPM Training ──")
trainer = DDPMTrainer(denoiser, ns)
ids     = torch.randint(0, V, (2, 8))
loss_simple = trainer.loss(ids, "simple")
loss_vlb    = trainer.loss(ids, "vlb")
print(f"  Simple loss: {loss_simple.item():.4f}")
print(f"  VLB loss:    {loss_vlb.item():.4f}")
print(f"  Loss at t=0: {trainer.loss_at_t(ids, 0):.4f}")
print(f"  Loss at t=99: {trainer.loss_at_t(ids, 99):.4f}")

# ── DDPM Sampling ─────────────────────────────────────────────────────────────
print("\n── DDPM Sampling (10 steps) ──")
ddpm_sampler = DDPMSampler(denoiser, ns)
samples = ddpm_sampler.sample(batch_size=2, seq_len=8, n_steps=10)
print(f"  Generated shape: {tuple(samples.shape)}")

# ── DDIM Sampling ─────────────────────────────────────────────────────────────
print("\n── DDIM Sampling (5 steps — fast!) ──")
ddim_sampler = DDIMSampler(denoiser, ns, eta=0.0)
samples_ddim = ddim_sampler.sample(batch_size=2, seq_len=8, n_steps=5)
print(f"  DDIM generated shape: {tuple(samples_ddim.shape)}")

# ── Masked Diffusion ──────────────────────────────────────────────────────────
print("\n── Masked Diffusion LM (D3PM style) ──")
mdlm   = MaskedDiffusionLM(vocab_size=V, d_model=32, n_layers=2)
ids    = torch.randint(2, V, (2, 8))   # avoid MASK_ID=1
masked, mask = mdlm.forward_mask(ids, t=50, T=100)
print(f"  Mask rate at t=50: {mask.float().mean():.2f}")
loss_m = mdlm.loss(ids, t=50, T=100)
print(f"  Masked diffusion loss: {loss_m.item():.4f}")
gen = mdlm.generate(batch_size=2, seq_len=8, T=5)
print(f"  Generated tokens: {tuple(gen.shape)}")

# ── Classifier-Free Guidance ──────────────────────────────────────────────────
print("\n── Classifier-Free Guidance ──")
cfg     = ClassifierFreeGuidance(denoiser, guidance_scale=7.5)
x_t_cfg = torch.randn(2, 8, 32)
t_cfg   = torch.randint(0, 100, (2,))
cond    = torch.randn(2, 32)
eps_cfg = cfg.guided_predict(x_t_cfg, t_cfg, cond)
print(f"  CFG eps shape: {tuple(eps_cfg.shape)}")
print(f"  Dynamic scale at t=50: {cfg.guidance_scale_schedule(50, 100):.2f}")

# ── DiffusionLMPipeline ───────────────────────────────────────────────────────
print("\n── DiffusionLMPipeline (continuous) ──")
pipe = DiffusionLMPipeline(vocab_size=V, d_model=32, n_layers=2, n_steps=50)
for _ in range(3):
    loss = pipe.train_step(torch.randint(0, V, (2, 8)))
print(f"  Training loss: {loss:.4f}")
tokens = pipe.generate(batch_size=2, seq_len=8, n_steps=5, sampler="ddim")
print(f"  Generated: {tuple(tokens.shape)}")
print(f"  Pipeline info: {pipe.info()}")

print("\n── DiffusionLMPipeline (masked) ──")
pipe_m = DiffusionLMPipeline(vocab_size=V, d_model=32, n_layers=2, mode="masked")
for _ in range(3):
    loss_m = pipe_m.train_step(torch.randint(2, V, (2, 8)))
print(f"  Masked training loss: {loss_m:.4f}")
tokens_m = pipe_m.generate(batch_size=2, seq_len=8, n_steps=5)
print(f"  Masked generated: {tuple(tokens_m.shape)}")
print("\nDiffusion LM demo complete!")
''')
commit("feat: add examples/diffusion_demo.py — schedule, denoiser, DDPM/DDIM, masked, CFG, pipeline")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_diffusion.py", '''\
"""tests/test_diffusion.py — Tests for NanoMind Diffusion Language Models."""
import pytest
import torch
from nanomind.diffusion import (
    NoiseSchedule, DiffusionDenoiser, SinusoidalTimeEmbedding,
    DDPMTrainer, DDPMSampler, DDIMSampler,
    MaskedDiffusionLM, MASK_ID, ClassifierFreeGuidance,
    DiffusionLMPipeline,
)

V = 16


# ── NoiseSchedule ─────────────────────────────────────────────────────────────

class TestNoiseSchedule:
    def _sched(self, sched="cosine"):
        return NoiseSchedule(n_steps=100, schedule=sched)

    def test_linear_schedule(self):
        ns = self._sched("linear")
        assert ns.betas.shape == (100,)
        assert ns.betas[0] < ns.betas[-1]

    def test_cosine_schedule(self):
        ns = self._sched("cosine")
        assert ns.alpha_bars[0] > ns.alpha_bars[-1]

    def test_sigmoid_schedule(self):
        ns = self._sched("sigmoid")
        assert ns.betas.shape == (100,)

    def test_alpha_bars_decreasing(self):
        ns = self._sched("cosine")
        # ᾱ should be monotonically decreasing
        assert (ns.alpha_bars[1:] <= ns.alpha_bars[:-1]).all()

    def test_q_sample_shape(self):
        ns  = self._sched()
        x0  = torch.randn(2, 4, 8)
        t   = torch.tensor([10, 50])
        x_t, noise = ns.q_sample(x0, t)
        assert x_t.shape == x0.shape
        assert noise.shape == x0.shape

    def test_snr_decreases(self):
        ns = self._sched()
        assert ns.snr(0) > ns.snr(50) > ns.snr(99)

    def test_predict_x0_shape(self):
        ns  = self._sched()
        x_t = torch.randn(2, 4, 8)
        eps = torch.randn(2, 4, 8)
        t   = torch.tensor([20, 40])
        x0  = ns.predict_x0(x_t, t, eps)
        assert x0.shape == x_t.shape

    def test_to_dict_keys(self):
        d = self._sched().to_dict()
        for k in ("n_steps", "schedule", "beta_min", "beta_max"):
            assert k in d


# ── SinusoidalTimeEmbedding ───────────────────────────────────────────────────

class TestSinusoidalTimeEmbedding:
    def test_output_shape(self):
        te = SinusoidalTimeEmbedding(32)
        t  = torch.tensor([0, 50, 99])
        assert te(t).shape == (3, 32)

    def test_different_timesteps_different_emb(self):
        te = SinusoidalTimeEmbedding(32)
        e0 = te(torch.tensor([0]))
        e50 = te(torch.tensor([50]))
        assert not torch.allclose(e0, e50)


# ── DiffusionDenoiser ─────────────────────────────────────────────────────────

class TestDiffusionDenoiser:
    def _denoiser(self):
        return DiffusionDenoiser(V, d_model=16, n_layers=1, n_heads=2, max_seq=8)

    def test_forward_shape(self):
        d   = self._denoiser()
        x   = torch.randn(2, 4, 16)
        t   = torch.randint(0, 100, (2,))
        out = d(x, t)
        assert out.shape == (2, 4, 16)

    def test_embed_tokens_shape(self):
        d   = self._denoiser()
        ids = torch.randint(0, V, (2, 4))
        emb = d.embed_tokens(ids)
        assert emb.shape == (2, 4, 16)

    def test_decode_to_tokens_shape(self):
        d   = self._denoiser()
        emb = torch.randn(2, 4, 16)
        ids = d.decode_to_tokens(emb)
        assert ids.shape == (2, 4)

    def test_decode_ids_in_vocab(self):
        d   = self._denoiser()
        emb = torch.randn(2, 4, 16)
        ids = d.decode_to_tokens(emb)
        assert (ids >= 0).all() and (ids < V).all()

    def test_n_params_positive(self):
        d = self._denoiser()
        assert d.n_params > 0


# ── DDPMTrainer ───────────────────────────────────────────────────────────────

class TestDDPMTrainer:
    def _setup(self):
        ns = NoiseSchedule(n_steps=50, schedule="cosine")
        d  = DiffusionDenoiser(V, d_model=16, n_layers=1, n_heads=2, max_seq=8)
        return DDPMTrainer(d, ns), d, ns

    def test_loss_returns_scalar(self):
        t, d, ns = self._setup()
        ids      = torch.randint(0, V, (2, 4))
        loss     = t.loss(ids)
        assert loss.shape == ()

    def test_loss_positive(self):
        t, d, ns = self._setup()
        ids      = torch.randint(0, V, (2, 4))
        assert t.loss(ids).item() > 0

    def test_vlb_loss(self):
        t, d, ns = self._setup()
        ids      = torch.randint(0, V, (2, 4))
        loss     = t.loss(ids, "vlb")
        assert loss.item() >= 0

    def test_loss_at_t(self):
        t, d, ns = self._setup()
        ids      = torch.randint(0, V, (2, 4))
        l        = t.loss_at_t(ids, 10)
        assert isinstance(l, float)

    def test_gradient_flows(self):
        t, d, ns = self._setup()
        ids      = torch.randint(0, V, (2, 4))
        loss     = t.loss(ids)
        loss.backward()
        has_grad = any(p.grad is not None for p in d.parameters())
        assert has_grad


# ── DDPMSampler ───────────────────────────────────────────────────────────────

class TestDDPMSampler:
    def _sampler(self):
        ns = NoiseSchedule(n_steps=20, schedule="cosine")
        d  = DiffusionDenoiser(V, d_model=16, n_layers=1, n_heads=2, max_seq=8)
        return DDPMSampler(d, ns)

    def test_sample_shape(self):
        s    = self._sampler()
        toks = s.sample(batch_size=2, seq_len=4, n_steps=3)
        assert toks.shape == (2, 4)

    def test_sample_ids_in_vocab(self):
        s    = self._sampler()
        toks = s.sample(batch_size=1, seq_len=4, n_steps=2)
        assert (toks >= 0).all() and (toks < V).all()


# ── DDIMSampler ───────────────────────────────────────────────────────────────

class TestDDIMSampler:
    def _sampler(self, eta=0.0):
        ns = NoiseSchedule(n_steps=20, schedule="cosine")
        d  = DiffusionDenoiser(V, d_model=16, n_layers=1, n_heads=2, max_seq=8)
        return DDIMSampler(d, ns, eta=eta)

    def test_sample_shape(self):
        s    = self._sampler()
        toks = s.sample(batch_size=2, seq_len=4, n_steps=5)
        assert toks.shape == (2, 4)

    def test_deterministic(self):
        """eta=0 → same seed → same output."""
        s = self._sampler(eta=0.0)
        torch.manual_seed(42)
        t1 = s.sample(batch_size=1, seq_len=4, n_steps=3)
        torch.manual_seed(42)
        t2 = s.sample(batch_size=1, seq_len=4, n_steps=3)
        assert torch.all(t1 == t2)

    def test_stochastic_eta1(self):
        """eta=1 should generally differ between runs."""
        s  = self._sampler(eta=1.0)
        t1 = s.sample(batch_size=1, seq_len=8, n_steps=3)
        t2 = s.sample(batch_size=1, seq_len=8, n_steps=3)
        assert t1.shape == t2.shape


# ── MaskedDiffusionLM ─────────────────────────────────────────────────────────

class TestMaskedDiffusionLM:
    def _mdlm(self):
        return MaskedDiffusionLM(vocab_size=V, d_model=16, n_layers=1, n_heads=2, max_seq=8)

    def test_forward_mask_rate(self):
        mdlm   = self._mdlm()
        ids    = torch.randint(2, V, (4, 6))
        masked, mask = mdlm.forward_mask(ids, t=50, T=100)
        rate   = mask.float().mean().item()
        assert 0.3 < rate < 0.7   # ≈ 0.5 at t=50

    def test_predict_shape(self):
        mdlm = self._mdlm()
        ids  = torch.randint(0, V, (2, 4))
        out  = mdlm.predict(ids, t=20)
        assert out.shape == (2, 4, V)

    def test_loss_scalar(self):
        mdlm = self._mdlm()
        ids  = torch.randint(2, V, (2, 4))
        loss = mdlm.loss(ids, t=50, T=100)
        assert loss.shape == ()

    def test_generate_shape(self):
        mdlm = self._mdlm()
        gen  = mdlm.generate(batch_size=2, seq_len=4, T=3)
        assert gen.shape == (2, 4)


# ── ClassifierFreeGuidance ────────────────────────────────────────────────────

class TestCFG:
    def _cfg(self):
        d   = DiffusionDenoiser(V, d_model=16, n_layers=1, n_heads=2, max_seq=8)
        return ClassifierFreeGuidance(d, guidance_scale=5.0)

    def test_guided_predict_shape(self):
        cfg  = self._cfg()
        x_t  = torch.randn(2, 4, 16)
        t    = torch.randint(0, 100, (2,))
        cond = torch.randn(2, 16)
        eps  = cfg.guided_predict(x_t, t, cond)
        assert eps.shape == (2, 4, 16)

    def test_train_mask_zeros_some(self):
        cfg   = ClassifierFreeGuidance(None, p_uncond=0.5)
        cond  = torch.randn(100, 16)
        mc    = cfg.train_mask(cond)
        # ~50% should be zeroed
        zeroed = (mc == 0).all(dim=-1).float().mean()
        assert 0.3 < zeroed < 0.7

    def test_guidance_scale_schedule_range(self):
        cfg = ClassifierFreeGuidance(None, guidance_scale=7.5)
        s0  = cfg.guidance_scale_schedule(0, 100)
        s50 = cfg.guidance_scale_schedule(50, 100)
        s99 = cfg.guidance_scale_schedule(99, 100)
        assert s0 >= s50 >= s99   # higher guidance at low noise


# ── DiffusionLMPipeline ───────────────────────────────────────────────────────

class TestDiffusionLMPipeline:
    def _pipe(self, mode="continuous"):
        return DiffusionLMPipeline(V, d_model=16, n_layers=1, n_steps=20, mode=mode)

    def test_continuous_train_step(self):
        p    = self._pipe("continuous")
        ids  = torch.randint(0, V, (2, 4))
        loss = p.train_step(ids)
        assert isinstance(loss, float)

    def test_masked_train_step(self):
        p    = self._pipe("masked")
        ids  = torch.randint(2, V, (2, 4))
        loss = p.train_step(ids)
        assert isinstance(loss, float)

    def test_generate_ddim(self):
        p    = self._pipe("continuous")
        toks = p.generate(batch_size=2, seq_len=4, n_steps=3, sampler="ddim")
        assert toks.shape == (2, 4)

    def test_generate_masked(self):
        p    = self._pipe("masked")
        toks = p.generate(batch_size=2, seq_len=4, n_steps=3)
        assert toks.shape == (2, 4)

    def test_info_keys(self):
        p = self._pipe("continuous")
        d = p.info()
        for k in ("mode", "n_steps", "n_params"):
            assert k in d
''')
commit("test: add full diffusion test suite — schedule, denoiser, DDPM/DDIM, masked, CFG, pipeline")

# COMMITS 12-18
for title, body in [
    ("test: add NoiseSchedule alpha_bars at T is near zero test", '''
class TestAlphaBarsLimits:
    def test_alpha_bar_t0_near_1(self):
        ns = NoiseSchedule(n_steps=100, schedule="cosine")
        assert ns.alpha_bars[0].item() > 0.9

    def test_alpha_bar_tT_near_0(self):
        ns = NoiseSchedule(n_steps=100, schedule="cosine")
        assert ns.alpha_bars[-1].item() < 0.1
'''),
    ("test: add q_sample with fixed noise is deterministic test", '''
class TestQSampleDeterministic:
    def test_fixed_noise(self):
        ns    = NoiseSchedule(n_steps=50, schedule="cosine")
        x0    = torch.randn(2, 4, 8)
        t     = torch.tensor([10, 30])
        noise = torch.randn_like(x0)
        x1, _ = ns.q_sample(x0, t, noise=noise)
        x2, _ = ns.q_sample(x0, t, noise=noise)
        assert torch.allclose(x1, x2)
'''),
    ("test: add DiffusionDenoiser AdaLN blocks run test", '''
class TestDenoiserBlocks:
    def test_n_blocks_matches_layers(self):
        d = DiffusionDenoiser(V, d_model=16, n_layers=3, n_heads=2, max_seq=8)
        assert len(d.blocks) == 3

    def test_different_timesteps_different_output(self):
        d = DiffusionDenoiser(V, d_model=16, n_layers=1, n_heads=2, max_seq=8)
        x = torch.randn(1, 4, 16)
        t0  = torch.tensor([0])
        t99 = torch.tensor([99])
        with torch.no_grad():
            o0  = d(x, t0)
            o99 = d(x, t99)
        assert not torch.allclose(o0, o99)
'''),
    ("test: add DDIMSampler faster than DDPM sampling test", '''
class TestDDIMFast:
    def test_ddim_fewer_steps_same_shape(self):
        ns = NoiseSchedule(n_steps=100, schedule="cosine")
        d  = DiffusionDenoiser(V, d_model=16, n_layers=1, n_heads=2, max_seq=8)
        ddim = DDIMSampler(d, ns, eta=0.0)
        # 10 steps instead of 100
        toks = ddim.sample(batch_size=1, seq_len=4, n_steps=10)
        assert toks.shape == (1, 4)
'''),
    ("test: add MaskedDiffusionLM fully masked at t=T test", '''
class TestMaskedAtTmax:
    def test_all_masked_at_t_eq_T(self):
        mdlm = MaskedDiffusionLM(vocab_size=V, d_model=16, n_layers=1)
        ids  = torch.randint(2, V, (2, 8))
        _, mask = mdlm.forward_mask(ids, t=100, T=100)
        # At t=T all should be masked
        assert mask.all()

    def test_none_masked_at_t_0(self):
        mdlm = MaskedDiffusionLM(vocab_size=V, d_model=16, n_layers=1)
        ids  = torch.randint(2, V, (2, 8))
        _, mask = mdlm.forward_mask(ids, t=0, T=100)
        assert not mask.any()
'''),
    ("test: add DiffusionPipeline info dict has schedule test", '''
class TestPipelineInfo:
    def test_continuous_info_has_schedule(self):
        p = DiffusionLMPipeline(V, d_model=16, n_layers=1, n_steps=20)
        d = p.info()
        assert "schedule" in d
        assert d["mode"] == "continuous"

    def test_masked_info_no_schedule(self):
        p = DiffusionLMPipeline(V, d_model=16, n_layers=1, mode="masked")
        d = p.info()
        assert d["mode"] == "masked"
'''),
    ("test: add CFG scale larger means larger diff test", '''
class TestCFGScale:
    def test_higher_scale_changes_prediction(self):
        d    = DiffusionDenoiser(V, d_model=16, n_layers=1, n_heads=2, max_seq=8)
        x_t  = torch.randn(2, 4, 16)
        t    = torch.randint(0, 50, (2,))
        cond = torch.randn(2, 16)
        cfg1 = ClassifierFreeGuidance(d, guidance_scale=1.0)
        cfg5 = ClassifierFreeGuidance(d, guidance_scale=5.0)
        e1   = cfg1.guided_predict(x_t, t, cond)
        e5   = cfg5.guided_predict(x_t, t, cond)
        # Different scales → different predictions
        assert not torch.allclose(e1, e5)
'''),
]:
    src = read("tests/test_diffusion.py")
    src += "\n" + body
    write("tests/test_diffusion.py", src)
    commit(title)

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v4.3.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"4.2.0\"", "__version__ = \"4.3.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v4.3.0 — Diffusion Language Models release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `gnn`        | Graph Neural Networks — AST parser, GCN/GAT/GGNN, code clone detection |",
    "| `gnn`        | Graph Neural Networks — AST parser, GCN/GAT/GGNN, code clone detection |\n"
    "| `diffusion`  | Diffusion LMs — DDPM/DDIM, masked diffusion, CFG, cosine/linear schedules |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [4.3.0] — 2024 — Diffusion Language Models\n\n### Added\n"
      "- `NoiseSchedule` — linear/cosine/sigmoid β, q_sample, snr, predict_x0\n"
      "- `SinusoidalTimeEmbedding` — DDPM-style sinusoidal timestep encoding\n"
      "- `DiffusionTransformerBlock` — AdaLN-conditioned transformer block\n"
      "- `DiffusionDenoiser` — AdaLN transformer, eps prediction, decode_to_tokens\n"
      "- `DDPMTrainer` — simple/VLB noise prediction loss, loss_at_t\n"
      "- `DDPMSampler` — full ancestral DDPM sampling, p_sample_step\n"
      "- `DDIMSampler` — fast 50-step sampling, DDIM inversion, eta control\n"
      "- `MaskedDiffusionLM` — D3PM-style masked diffusion, iterative generate\n"
      "- `ClassifierFreeGuidance` — CFG interpolation, train_mask, dynamic scale\n"
      "- `DiffusionLMPipeline` — unified continuous/masked pipeline\n"
      "- `examples/diffusion_demo.py` — full diffusion LM demo\n\n---\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v4.3.0, update README and CHANGELOG for Day 43 Diffusion LMs")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 43 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v4.3.0",
    "-m", "NanoMind v4.3.0 — Diffusion Language Models", check=False)
r = run("git", "push", "origin", "v4.3.0", check=False)
print("Tag v4.3.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 43 COMPLETE — v4.3.0 TAGGED! ===")
