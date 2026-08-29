"""
nanomind/logging/console.py — Rich terminal logger for training progress.
"""

from __future__ import annotations

import sys
from nanomind.logging.base import BaseLogger
from nanomind.utils.logger import get_logger

_log = get_logger("training")


class ConsoleLogger(BaseLogger):
    """
    Console (stdout) logger — always available, no external dependencies.

    Formats training metrics as a clean one-liner per log step::

        step  250 | loss 2.3451 | lr 3.00e-04 | grad_norm 1.23 | tok/s 12540

    Args:
        run_name:     Name displayed in the header.
        log_interval: Print frequency (steps).
    """

    def __init__(self, run_name: str = "run", log_interval: int = 50) -> None:
        self.run_name     = run_name
        self.log_interval = log_interval
        self._config      = {}

    def log_config(self, config: dict) -> None:
        print("=" * 60)
        print(f"  NanoMind Training Run: {self.run_name}")
        print("=" * 60)
        for k, v in config.items():
            print(f"  {k:<20}: {v}")
        print("=" * 60)
        self._config = config

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        parts = [f"step {step:>6}"]
        for key, val in metrics.items():
            if isinstance(val, float):
                if "loss" in key or "ppl" in key:
                    parts.append(f"{key} {val:.4f}")
                elif "lr" in key:
                    parts.append(f"{key} {val:.2e}")
                elif "norm" in key:
                    parts.append(f"{key} {val:.3f}")
                elif "tok" in key or "rate" in key:
                    parts.append(f"{key} {val:.0f}")
                else:
                    parts.append(f"{key} {val:.4f}")
            else:
                parts.append(f"{key} {val}")
        print(" | ".join(parts), flush=True)

    def log_histogram(self, name: str, values, step: int) -> None:
        pass  # Console logger skips histograms

    def log_text(self, tag: str, text: str, step: int) -> None:
        print(f"[{step}] {tag}: {text[:120]}")

    def finish(self) -> None:
        print("=" * 60)
        print(f"  Training complete: {self.run_name}")
        print("=" * 60)
