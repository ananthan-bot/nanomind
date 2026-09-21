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
