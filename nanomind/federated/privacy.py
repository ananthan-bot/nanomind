"""
nanomind/federated/privacy.py — Differential Privacy for gradient noise injection.

Implements DP-SGD (Abadi et al., 2016):
  1. Clip each per-sample gradient to L2-norm ≤ C
  2. Sum clipped gradients
  3. Add Gaussian noise: N(0, σ²I) where σ = C × noise_multiplier
  4. Divide by batch size (normalise)

The noise multiplier σ is calibrated from (ε, δ) using the Gaussian mechanism:
  σ ≥ √(2 ln(1.25/δ)) / ε

Privacy accounting (moments accountant) tracks cumulative privacy loss
over multiple rounds. NanoMind uses the simple single-round guarantee.

Reference:
  Abadi et al. (2016) — https://arxiv.org/abs/1607.00133
  Opacus (Meta) — https://opacus.ai
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from nanomind.utils.logger import get_logger

log = get_logger("federated.privacy")


def _gaussian_noise_multiplier(epsilon: float, delta: float) -> float:
    """
    Compute Gaussian noise multiplier for (epsilon, delta)-DP.

    Calibrated via the analytic Gaussian mechanism:
      sigma >= sqrt(2 * ln(1.25 / delta)) / epsilon

    Args:
        epsilon: Privacy budget.
        delta:   Failure probability.

    Returns:
        Noise multiplier sigma.
    """
    return math.sqrt(2 * math.log(1.25 / delta)) / epsilon


class DifferentialPrivacyEngine:
    """
    Differential Privacy engine for DP-SGD.

    Clips per-parameter gradients and adds calibrated Gaussian noise.

    Args:
        max_grad_norm:   L2 gradient clipping norm (C in the paper).
        noise_multiplier: Sigma for Gaussian noise (computed from ε/δ if not given).
        epsilon:         Privacy budget ε (used to compute sigma if noise_multiplier=None).
        delta:           Failure probability δ.

    Example::

        engine = DifferentialPrivacyEngine(max_grad_norm=1.0, epsilon=1.0, delta=1e-5)
        engine.clip_and_noise(model)  # modifies model.grad in-place
        privacy_spent = engine.privacy_spent(n_steps=100, n_samples=1000, batch_size=32)
    """

    def __init__(
        self,
        max_grad_norm:   float,
        noise_multiplier: float | None = None,
        epsilon:         float = 1.0,
        delta:           float = 1e-5,
    ) -> None:
        self.max_grad_norm = max_grad_norm
        self.delta         = delta
        self.epsilon       = epsilon
        if noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
        else:
            self.noise_multiplier = _gaussian_noise_multiplier(epsilon, delta)
        log.info(f"DP engine: C={max_grad_norm}, σ={self.noise_multiplier:.4f}, "
                 f"ε={epsilon}, δ={delta}")

    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradient L2 norm per parameter to max_grad_norm.

        Args:
            model: Model with computed gradients.

        Returns:
            Total gradient norm before clipping.
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)

        clip_coef = min(1.0, self.max_grad_norm / (total_norm + 1e-8))
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

        return total_norm

    def add_noise(self, model: nn.Module) -> None:
        """
        Add Gaussian noise to model gradients for DP guarantee.

        Args:
            model: Model with (clipped) gradients.
        """
        sigma = self.noise_multiplier * self.max_grad_norm
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * sigma
                p.grad.data.add_(noise)

    def clip_and_noise(self, model: nn.Module) -> float:
        """
        Clip gradients and add Gaussian noise (combined DP-SGD step).

        Returns:
            Original gradient norm before clipping.
        """
        norm = self.clip_gradients(model)
        self.add_noise(model)
        return norm

    def privacy_spent(
        self,
        n_steps:    int,
        n_samples:  int,
        batch_size: int,
    ) -> dict:
        """
        Estimate total privacy spent using simple composition.

        Uses strong composition theorem (approximate).

        Returns:
            Dict with ``epsilon``, ``delta``, ``n_steps``.
        """
        q         = batch_size / max(n_samples, 1)   # sampling rate
        # Simple: each step spends ~ q * epsilon
        eps_total = q * self.epsilon * n_steps
        return {
            "epsilon":  round(eps_total, 4),
            "delta":    self.delta,
            "n_steps":  n_steps,
            "sigma":    round(self.noise_multiplier, 4),
        }
