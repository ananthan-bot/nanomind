"""
nanomind/continual/si.py — Synaptic Intelligence (SI).

SI (Zenke et al., 2017) computes parameter importance ONLINE during
training, rather than computing Fisher after training like EWC.

Importance is accumulated as the integral of the gradient × weight change:
  Ω_i = Σ_t (-g_i^t × Δθ_i^t) / ((Δθ_i)^2 + ξ)

High Ω_i → parameter contributed to loss reduction → protect it.

Advantages over EWC:
  - No extra backward pass after task completion
  - Computes importance continuously during training
  - Lower memory (no Fisher matrix storage per task)

Reference:
  Zenke, Poole & Ganguli (2017) "Continual Learning Through Synaptic Intelligence"
  https://arxiv.org/abs/1703.04200
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.utils.logger import get_logger

log = get_logger("continual.si")


class SynapticIntelligence:
    """
    Synaptic Intelligence (SI) continual learning regulariser.

    Args:
        model:  Language model.
        lambda_: Regularisation strength.
        xi:      Damping term for numerical stability.

    Example::

        si = SynapticIntelligence(model, lambda_=100.0)
        # Register parameter state before task begins
        si.begin_task()
        # During each training step:
        loss.backward()
        si.update_importances()   # call BEFORE optimiser step
        optimiser.step()
        # After task ends:
        si.end_task()
    """

    def __init__(
        self,
        model:   nn.Module,
        lambda_: float = 100.0,
        xi:      float = 1e-3,
    ) -> None:
        self.model   = model
        self.lambda_ = lambda_
        self.xi      = xi
        self._W:      dict = {}    # accumulated gradient × delta
        self._p_prev: dict = {}    # weights at task start
        self._p_old:  dict = {}    # weights at previous step
        self._omega:  list[dict] = []   # consolidated importances per task
        self._theta_star: list[dict] = []  # optimal weights per task

    def begin_task(self) -> None:
        """Call at the start of each new task."""
        self._W     = {n: torch.zeros_like(p.data)
                       for n, p in self.model.named_parameters() if p.requires_grad}
        self._p_prev = {n: p.data.clone()
                        for n, p in self.model.named_parameters() if p.requires_grad}
        self._p_old  = {n: p.data.clone()
                        for n, p in self.model.named_parameters() if p.requires_grad}

    def update_importances(self) -> None:
        """
        Update online importance estimates after backward pass.

        Call BEFORE the optimiser step so that current gradients
        and previous weights are both available.
        """
        for n, p in self.model.named_parameters():
            if p.grad is not None and n in self._W:
                delta = p.data - self._p_old[n]
                self._W[n] += -p.grad.data * delta
                self._p_old[n] = p.data.clone()

    def end_task(self) -> None:
        """Consolidate importances at end of task."""
        omega = {}
        for n, p in self.model.named_parameters():
            if n in self._W:
                delta_sq = (p.data - self._p_prev[n]).pow(2)
                omega[n] = self._W[n] / (delta_sq + self.xi)
                omega[n] = omega[n].abs()
        self._omega.append(omega)
        self._theta_star.append(
            {n: p.data.clone() for n, p in self.model.named_parameters()
             if p.requires_grad}
        )

    def penalty(self) -> torch.Tensor:
        """Compute SI regularisation penalty."""
        if not self._omega:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for omega, theta in zip(self._omega, self._theta_star):
            for n, p in self.model.named_parameters():
                if n in omega and p.requires_grad:
                    loss += (omega[n] * (p - theta[n]).pow(2)).sum()
        return (self.lambda_ / 2) * loss

    @property
    def n_tasks(self) -> int:
        return len(self._omega)
