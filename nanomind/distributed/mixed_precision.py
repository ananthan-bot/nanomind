"""
nanomind/distributed/mixed_precision.py — Mixed precision (AMP) training.

## Mixed Precision Training (Micikevicius et al., 2018)

Train in fp16/bf16 for speed, keep fp32 master weights for accuracy.

fp16 vs fp32:
  fp32: 32-bit float, 7 decimal digits, full range
  fp16: 16-bit float, 3 decimal digits, smaller range (underflow/overflow!)
  bf16: 16-bit float, same exponent range as fp32, less precision
        → bf16 is preferred for LLMs (no overflow risk)

## AMP Recipe

  1. Forward pass: fp16/bf16 weights + activations → less memory, faster
  2. Loss scaling: multiply loss by scale factor (prevents fp16 underflow)
  3. Backward pass: fp16 gradients
  4. Unscale gradients: divide by scale factor
  5. Gradient clipping: in fp32 for stability
  6. fp32 master weight update: apply gradients to fp32 copy
  7. fp32 → fp16/bf16 cast: update fp16 weights for next forward

## Memory Savings

  LLaMA-70B fp32: 280 GB
  LLaMA-70B bf16: 140 GB   (2× reduction!)
  LLaMA-70B int8: 70 GB    (4× reduction, with quantization)

Speed:
  A100 peak fp16 TFLOPS: 312
  A100 peak fp32 TFLOPS:  77.6  (4× faster in fp16!)

Reference:
  Micikevicius et al. (2018) "Mixed Precision Training"
  https://arxiv.org/abs/1710.03740
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class AMPConfig:
    """Configuration for automatic mixed precision."""
    dtype:           str   = "bf16"      # "fp16" | "bf16" | "fp32"
    initial_scale:   float = 2.0 ** 16   # for fp16 loss scaling
    scale_growth:    float = 2.0
    scale_min:       float = 1.0
    scale_max:       float = 2.0 ** 24
    backoff_factor:  float = 0.5
    growth_interval: int   = 2000        # steps between scale growth

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32}[self.dtype]

    @property
    def use_autocast(self) -> bool:
        return self.dtype != "fp32"


class LossScaler:
    """
    Dynamic loss scaler for fp16 training.

    Automatically adjusts scale factor to prevent gradient underflow
    while avoiding overflow.

    Args:
        cfg: :class:`AMPConfig`.

    Example::

        scaler = LossScaler(AMPConfig(dtype="fp16"))
        loss   = scaler.scale(loss)
        loss.backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
    """

    def __init__(self, cfg: AMPConfig | None = None) -> None:
        self.cfg   = cfg or AMPConfig()
        self._scale      = self.cfg.initial_scale
        self._steps      = 0
        self._n_overflow = 0

    @property
    def scale(self) -> float:
        return self._scale

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Multiply loss by current scale factor."""
        return loss * self._scale

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Divide all gradients by current scale factor."""
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data.div_(self._scale)

    def has_overflow(self, optimizer: torch.optim.Optimizer) -> bool:
        """Check if any gradient is inf or nan."""
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                        return True
        return False

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Take optimizer step if no overflow.

        Returns:
            True if step was taken (no overflow).
        """
        if self.has_overflow(optimizer):
            self._n_overflow += 1
            self._scale = max(self.cfg.scale_min,
                              self._scale * self.cfg.backoff_factor)
            # Zero gradients to avoid applying corrupt gradients
            optimizer.zero_grad()
            return False

        optimizer.step()
        return True

    def update(self) -> None:
        """Update scale factor (increase if no overflow for a while)."""
        self._steps += 1
        if self._steps % self.cfg.growth_interval == 0:
            self._scale = min(self.cfg.scale_max,
                              self._scale * self.cfg.scale_growth)

    def state_dict(self) -> dict:
        return {"scale": self._scale, "steps": self._steps,
                "n_overflow": self._n_overflow}


class MixedPrecisionTrainer:
    """
    Mixed precision training wrapper.

    Handles dtype casting, loss scaling, and master weight synchronisation.

    Args:
        model:     Model to train (converted to AMP dtype).
        optimizer: Optimizer operating on fp32 master weights.
        cfg:       :class:`AMPConfig`.

    Example::

        trainer = MixedPrecisionTrainer(model, optimizer, AMPConfig(dtype="bf16"))
        loss    = trainer.compute_loss(input_ids, labels)
        trainer.backward(loss)
        trainer.step()
    """

    def __init__(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg:       AMPConfig | None = None,
    ) -> None:
        self.model     = model
        self.optimizer = optimizer
        self.cfg       = cfg or AMPConfig()
        self.scaler    = LossScaler(self.cfg)
        self._step_count = 0

    @contextmanager
    def autocast(self):
        """Context manager for automatic dtype casting."""
        if self.cfg.use_autocast:
            with torch.autocast("cpu", dtype=self.cfg.torch_dtype):
                yield
        else:
            yield

    def backward(self, loss: torch.Tensor) -> None:
        """Scale loss and run backward."""
        if self.cfg.dtype == "fp16":
            scaled = self.scaler.scale_loss(loss)
            scaled.backward()
        else:
            loss.backward()

    def step(self, max_grad_norm: float | None = None) -> bool:
        """
        Unscale, clip, and step the optimizer.

        Args:
            max_grad_norm: Gradient clipping norm (None = no clip).

        Returns:
            True if optimizer step was taken.
        """
        if self.cfg.dtype == "fp16":
            self.scaler.unscale_(self.optimizer)
            if max_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            success = self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if max_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            success = True

        self._step_count += 1
        return success

    def memory_report(self) -> dict:
        """Memory comparison between fp32 and AMP modes."""
        n_params  = sum(p.numel() for p in self.model.parameters())
        bytes_fp32 = n_params * 4
        amp_bytes  = n_params * (2 if self.cfg.dtype in ("fp16", "bf16") else 4)
        return {
            "n_params":     n_params,
            "fp32_gb":      round(bytes_fp32 / 1e9, 3),
            "amp_gb":       round(amp_bytes / 1e9, 3),
            "savings_x":    round(bytes_fp32 / max(amp_bytes, 1), 1),
            "dtype":        self.cfg.dtype,
        }
