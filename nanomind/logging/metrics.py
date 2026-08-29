"""
nanomind/logging/metrics.py — Metrics accumulation and averaging buffer.
"""

from __future__ import annotations

from collections import defaultdict


class MetricsBuffer:
    """
    Accumulate scalar metrics over multiple steps and compute running averages.

    Useful for computing per-epoch or per-interval averages of training metrics
    without keeping the full history in memory.

    Example::

        buf = MetricsBuffer()
        for x, y in loader:
            loss = model(x, y)
            buf.update({"loss": loss.item()})
        print(buf.averages())   # {"loss": 2.345}
        buf.reset()
    """

    def __init__(self) -> None:
        self._sums:   dict[str, float] = defaultdict(float)
        self._counts: dict[str, int]   = defaultdict(int)

    def update(self, metrics: dict[str, float], n: int = 1) -> None:
        """
        Add metric values to the buffer.

        Args:
            metrics: Dict of metric_name → value.
            n:       Number of samples these metrics are averaged over.
        """
        for k, v in metrics.items():
            self._sums[k]   += v * n
            self._counts[k] += n

    def averages(self) -> dict[str, float]:
        """
        Compute the running average for each accumulated metric.

        Returns:
            Dict of metric_name → average_value.
        """
        return {
            k: self._sums[k] / max(self._counts[k], 1)
            for k in self._sums
        }

    def reset(self) -> None:
        """Clear all accumulated values."""
        self._sums.clear()
        self._counts.clear()

    def __len__(self) -> int:
        return len(self._sums)

    def __contains__(self, key: str) -> bool:
        return key in self._sums
