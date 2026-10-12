"""
nanomind/flow/schedules.py — Noise schedules for discrete diffusion (cosine, linear, absorbing rates).
"""
import math
from typing import Dict, Any, Tuple
import torch


class CosineNoiseSchedule:
    """
    Cosine noise schedule for discrete text diffusion (Nichol & Dhariwal):
    alpha_bar(t) = cos^2(((t/T + s) / (1 + s)) * pi / 2)
    Prevents abrupt corruption near t=0 and t=T.
    """

    def __init__(self, num_timesteps: int = 20, s: float = 0.008):
        self.num_timesteps = num_timesteps
        self.s = s

        # Precompute alpha_bars for t in [0, num_timesteps]
        steps = torch.arange(num_timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((steps / num_timesteps + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2
        alpha_bars = f_t / f_t[0]
        self.alpha_bars = torch.clamp(alpha_bars, min=1e-5, max=1.0)

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Returns alpha_bar (fraction of uncorrupted signal) at timestep t."""
        return self.alpha_bars[t.long()]

    def sample_corrupted_tokens(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        mask_token_id: int = 0,
    ) -> torch.Tensor:
        """
        Applies absorbing-state corruption at timestep t:
        With probability (1 - alpha_bar(t)), token is replaced by [MASK].
        x_0: (B, T_seq)
        t: (B,)
        Returns: x_t: (B, T_seq)
        """
        B, T_seq = x_0.shape
        alpha_bar = self.get_alpha_bar(t).to(x_0.device).unsqueeze(1)  # (B, 1)

        # Draw uniform random mask: mask where rand > alpha_bar
        rand = torch.rand(B, T_seq, device=x_0.device)
        is_masked = rand > alpha_bar

        x_t = torch.where(is_masked, torch.tensor(mask_token_id, device=x_0.device), x_0)
        return x_t


class LinearNoiseSchedule:
    """Standard linear beta schedule for discrete diffusion."""

    def __init__(self, num_timesteps: int = 20, beta_start: float = 1e-4, beta_end: float = 0.2):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_bars[t.long()]
