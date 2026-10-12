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
