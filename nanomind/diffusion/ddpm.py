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
