"""
nanomind/sae/steer.py — Feature steering and concept intervention using SAE dictionary directions.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class FeatureSteerer:
    """
    Steers model activations along interpretable SAE dictionary directions:
    h_steered = h + clamp_val * W_dec[:, feature_idx]
    """
    def __init__(self, sae: nn.Module):
        self.sae = sae

    def get_feature_direction(self, feature_idx: int) -> torch.Tensor:
        """Extract unit direction vector for an SAE dictionary atom."""
        # W_dec weight has shape (d_in, d_sae)
        return self.sae.w_dec.weight[:, feature_idx].detach()

    def steer_activation(
        self,
        activations: torch.Tensor,
        feature_idx: int,
        coefficient: float = 5.0,
    ) -> torch.Tensor:
        """
        Add scaled feature direction to input activations.
        activations: shape (..., d_in)
        """
        direction = self.get_feature_direction(feature_idx).to(activations.device)
        return activations + coefficient * direction

    def clamp_feature(
        self,
        activations: torch.Tensor,
        feature_idx: int,
        target_value: float = 10.0,
    ) -> torch.Tensor:
        """Encode activations, set target feature to fixed value, and decode."""
        latents = self.sae.encode(activations)
        latents[..., feature_idx] = target_value
        return self.sae.decode(latents)

    def suppress_feature(
        self,
        activations: torch.Tensor,
        feature_idx: int,
    ) -> torch.Tensor:
        """Ablate a specific feature atom (set latent activation to 0)."""
        return self.clamp_feature(activations, feature_idx, target_value=0.0)
