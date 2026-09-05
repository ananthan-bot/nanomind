"""
nanomind/amp/scaler.py — Gradient scaler for float16 AMP training.

float16 has a much smaller dynamic range than float32. During backward pass,
gradients can underflow to zero (too small for float16) or overflow to inf.

The GradScaler solution:
  1. Before backward: multiply loss by a large scale factor S
  2. During backward: gradients are S× larger → no underflow
  3. Before optimizer step: divide gradients by S → restore true scale
  4. Check for inf/nan: if detected, skip optimizer step and reduce S
  5. If several consecutive clean steps: increase S

This keeps gradients in the representable float16 range without changing
the training math (the scale cancels out in the optimizer step).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from nanomind.amp.config import AMPConfig
from nanomind.utils.logger import get_logger

log = get_logger("amp.scaler")


class NanoGradScaler:
    """
    Gradient scaler for float16 AMP training.

    Thin wrapper around ``torch.amp.GradScaler`` with NanoMind config integration.
    Automatically disabled when not on CUDA or when dtype is bfloat16.

    Args:
        cfg: AMP configuration.

    Example::

        scaler = NanoGradScaler(AMPConfig(dtype="float16"))
        with mixed_precision_context(cfg, "cuda"):
            loss = model(x, y)[1]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    """

    def __init__(self, cfg: AMPConfig) -> None:
        self.cfg     = cfg
        self._active = cfg.enabled and cfg.grad_scaler and torch.cuda.is_available()

        if self._active:
            self._scaler = torch.amp.GradScaler(
                "cuda",
                init_scale=cfg.init_scale,
                growth_interval=cfg.growth_interval,
            )
            log.info(f"GradScaler enabled (init_scale={cfg.init_scale:.0f})")
        else:
            self._scaler = None
            reason = "bfloat16" if cfg.dtype == "bfloat16" else "no CUDA"
            log.info(f"GradScaler disabled ({reason})")

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale the loss before backward."""
        if self._active:
            return self._scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients and call optimizer.step() (skips if inf/nan)."""
        if self._active:
            self._scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        """Update the scale factor for the next iteration."""
        if self._active:
            self._scaler.update()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Manually unscale gradients (needed before grad clipping)."""
        if self._active:
            self._scaler.unscale_(optimizer)

    @property
    def scale_factor(self) -> float:
        if self._active:
            return self._scaler.get_scale()
        return 1.0

    def state_dict(self) -> dict:
        if self._active:
            return self._scaler.state_dict()
        return {}

    def load_state_dict(self, state: dict) -> None:
        if self._active and state:
            self._scaler.load_state_dict(state)
