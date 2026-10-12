"""
nanomind/flow/sampler.py — Non-autoregressive parallel discrete diffusion and continuous flow samplers.
"""
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn

from nanomind.flow.config import FlowConfig
from nanomind.flow.d3pm import D3PM
from nanomind.flow.model import FlowTransformer


class DiscreteDiffusionSampler:
    """
    Non-autoregressive parallel sampler:
    Initializes sequence entirely as [MASK] tokens, then unmasks in parallel across K diffusion steps.
    """

    def __init__(self, model: FlowTransformer, config: Optional[FlowConfig] = None):
        self.model = model
        self.config = config or FlowConfig()
        self.d3pm = D3PM(
            vocab_size=self.config.vocab_size,
            mask_token_id=self.config.mask_token_id,
            num_timesteps=self.config.diffusion_steps,
        )

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len: int = 32,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generates full sequence in parallel in K diffusion steps.
        Returns: final_tokens: (B, seq_len)
        """
        self.model.eval()
        dev = device or next(self.model.parameters()).device

        # Step 1: Start completely masked: x_T = [MASK, MASK, ..., MASK]
        x_t = torch.full((batch_size, seq_len), self.config.mask_token_id, dtype=torch.long, device=dev)

        # Step 2: Denoise backwards from t = K-1 down to 0
        for step in reversed(range(self.config.diffusion_steps)):
            t_tensor = torch.full((batch_size,), step, dtype=torch.long, device=dev)
            logits = self.model(x_t, t_tensor)
            x_t = self.d3pm.compute_posterior_step(x_t, logits, step)

        return x_t


class FlowODEIntegrator:
    """
    Euler and Midpoint ODE solver for continuous probability flow matching:
    dx_t = v_theta(x_t, t) dt
    Integrates from t = 0 (noise) to t = 1 (clean latent representations).
    """

    def __init__(self, num_steps: int = 20):
        self.num_steps = num_steps

    def integrate_euler(
        self,
        vector_field_fn,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Euler forward integration:
        x_{t + dt} = x_t + dt * v_theta(x_t, t)
        """
        dt = 1.0 / self.num_steps
        x_t = x_0.clone()

        for i in range(self.num_steps):
            t_scalar = i / self.num_steps
            t_tensor = torch.full((x_0.shape[0],), t_scalar, device=x_0.device)
            v = vector_field_fn(x_t, t_tensor)
            x_t = x_t + dt * v

        return x_t


def apply_classifier_free_guidance(
    cond_logits: torch.Tensor,
    uncond_logits: torch.Tensor,
    cfg_scale: float = 2.0,
) -> torch.Tensor:
    """
    Applies Classifier-Free Guidance (CFG):
    logits = uncond + cfg_scale * (cond - uncond)
    """
    return uncond_logits + cfg_scale * (cond_logits - uncond_logits)
