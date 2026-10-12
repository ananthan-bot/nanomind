"""
nanomind/flow/flow_matching.py — Optimal Transport Continuous Flow Matching (OT-CFM) for language embeddings.
Generates straight-line probability trajectories between standard Gaussian prior and token embeddings.
"""
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimalTransportFlowMatching(nn.Module):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM) in continuous latent space:
    x_t = (1 - (1 - sigma)*t) * x_0 + t * x_1
    Target velocity vector: u_t(x | x_0, x_1) = x_1 - (1 - sigma) * x_0
    """

    def __init__(self, sigma: float = 1e-4):
        super().__init__()
        self.sigma = sigma

    def sample_location_and_target(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_0: source noise ~ N(0, I) of shape (B, T, D)
        x_1: target token embeddings of shape (B, T, D)
        t: timestep in [0, 1] of shape (B, 1, 1)
        Returns:
            x_t: interpolated point along straight-line trajectory
            u_t: target velocity vector d/dt x_t
        """
        # Linear interpolation
        x_t = (1.0 - (1.0 - self.sigma) * t) * x_0 + t * x_1
        # Constant velocity vector along straight trajectory
        u_t = x_1 - (1.0 - self.sigma) * x_0
        return x_t, u_t

    def compute_loss(self, predicted_v: torch.Tensor, target_u: torch.Tensor) -> torch.Tensor:
        """Mean squared error between predicted vector field and ground-truth OT velocity."""
        return F.mse_loss(predicted_v, target_u)
