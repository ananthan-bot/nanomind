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
