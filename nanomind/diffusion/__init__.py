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
