"""
nanomind/flow/d3pm.py — Discrete Denoising Diffusion Probabilistic Models (D3PM).
Implements categorical transition matrices and posterior distribution q(x_{t-1} | x_t, x_0).
"""
import math
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.flow.schedules import CosineNoiseSchedule


class D3PM:
    """
    D3PM with absorbing state (Austin et al., NeurIPS 2021).
    Forward process transitions clean tokens into [MASK] absorbing state.
    Reverse process predicts clean token distribution p_theta(x_0 | x_t, t).
    """

    def __init__(self, vocab_size: int = 1000, mask_token_id: int = 0, num_timesteps: int = 20):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.num_timesteps = num_timesteps
        self.schedule = CosineNoiseSchedule(num_timesteps=num_timesteps)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x_t from forward absorbing state transition."""
        return self.schedule.sample_corrupted_tokens(x_0, t, self.mask_token_id)

    def compute_posterior_step(
        self,
        x_t: torch.Tensor,
        pred_x0_logits: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        Samples x_{t-1} given current state x_t and predicted clean token logits pred_x0_logits.
        Tokens that are NOT masked in x_t stay unchanged (absorbing property).
        Tokens that ARE masked in x_t sample unmasking from posterior categorical distribution.
        """
        if t == 0:
            return torch.argmax(pred_x0_logits, dim=-1)

        B, T_seq, V = pred_x0_logits.shape
        probs_x0 = F.softmax(pred_x0_logits, dim=-1)

        # Sample candidate clean tokens from predicted distribution
        flat_probs = probs_x0.view(-1, V)
        sampled_x0 = torch.multinomial(flat_probs, num_samples=1).view(B, T_seq)

        # Calculate probability of unmasking at step t -> t-1
        alpha_t = self.schedule.alpha_bars[t]
        alpha_prev = self.schedule.alpha_bars[t - 1]
        p_unmask = (alpha_prev - alpha_t) / max(1e-5, (1.0 - alpha_t))

        # Decide which masked tokens get filled in this step
        rand = torch.rand(B, T_seq, device=x_t.device)
        should_unmask = (x_t == self.mask_token_id) & (rand < p_unmask)

        x_prev = torch.where(should_unmask, sampled_x0, x_t)
        return x_prev
