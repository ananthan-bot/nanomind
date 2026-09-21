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
            x      = ab_n.sqrt() * (x - (1 - ab).sqrt() * eps) / (ab.sqrt() + 1e-8)                      + (1 - ab_n).sqrt() * eps
        return x
