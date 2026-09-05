"""
nanomind/amp/accumulation.py — Gradient accumulation for large effective batch sizes.

Gradient accumulation simulates training with a large batch size by accumulating
gradients from multiple small micro-batches before calling optimizer.step().

  Effective batch size = micro_batch_size × grad_accum_steps

This lets you train with an effective batch of 512 samples on a GPU that only
fits 32 samples per step:
  micro_batch = 32, accum_steps = 16 → effective batch = 512

Key requirement: divide the loss by grad_accum_steps so gradients are correctly
scaled (equivalent to the mean over the full effective batch, not the sum).
"""

from __future__ import annotations


class GradAccumulator:
    """
    Helper class to track gradient accumulation state.

    Tracks the current micro-batch index and tells you when to call
    optimizer.step() (every ``accum_steps`` micro-batches).

    Args:
        accum_steps: Number of micro-batches to accumulate before stepping.

    Example::

        acc = GradAccumulator(accum_steps=4)
        for x, y in loader:
            is_last = acc.should_step()
            loss    = model(x, y)[1] / acc.accum_steps

            if is_last:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                acc.reset()
            else:
                with model.no_sync():   # DDP: skip allreduce
                    loss.backward()
                acc.step()
    """

    def __init__(self, accum_steps: int = 1) -> None:
        assert accum_steps >= 1
        self.accum_steps = accum_steps
        self._count      = 0

    def step(self) -> None:
        """Advance the micro-batch counter."""
        self._count = (self._count + 1) % self.accum_steps

    def should_step(self) -> bool:
        """Return True if this is the last micro-batch in the accumulation window."""
        return (self._count + 1) % self.accum_steps == 0

    def reset(self) -> None:
        """Reset counter (call after optimizer.step())."""
        self._count = 0

    @property
    def current_step(self) -> int:
        return self._count

    @property
    def loss_scale(self) -> float:
        """Divide loss by this to get correctly scaled mean gradient."""
        return float(self.accum_steps)
