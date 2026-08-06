"""
nanomind/utils/timer.py — Lightweight timing utilities for benchmarking.

Provides a context manager and a decorator for measuring elapsed time,
useful for tracking iteration speed during training.
"""

import time
from contextlib import contextmanager
from typing import Generator, Optional


class Timer:
    """
    A simple wall-clock timer with lap support.

    Example::

        timer = Timer()
        timer.start()
        # ... do work ...
        elapsed = timer.stop()
        print(f"Took {elapsed:.2f}s")
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._laps: list[float] = []

    def start(self) -> "Timer":
        """Start (or restart) the timer."""
        self._start = time.perf_counter()
        return self

    def lap(self) -> float:
        """Record a lap and return elapsed time since last lap (or start)."""
        now = time.perf_counter()
        assert self._start is not None, "Call .start() before .lap()"
        base = self._laps[-1] if self._laps else self._start
        elapsed = now - base
        self._laps.append(now)
        return elapsed

    def stop(self) -> float:
        """Stop the timer and return total elapsed time in seconds."""
        assert self._start is not None, "Call .start() before .stop()"
        return time.perf_counter() - self._start

    def reset(self) -> "Timer":
        """Reset all state."""
        self._start = None
        self._laps = []
        return self

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds without stopping."""
        assert self._start is not None, "Call .start() first"
        return time.perf_counter() - self._start


@contextmanager
def timed(label: str = "", verbose: bool = True) -> Generator[Timer, None, None]:
    """
    Context manager that times a block and optionally prints the result.

    Args:
        label:   Optional description to print with the timing.
        verbose: If True, print the elapsed time after the block exits.

    Yields:
        A :class:`Timer` instance (already started).

    Example::

        with timed("data loading"):
            data = load_data()
        # prints: data loading: 1.23s
    """
    t = Timer().start()
    try:
        yield t
    finally:
        elapsed = t.stop()
        if verbose:
            prefix = f"{label}: " if label else ""
            print(f"{prefix}{elapsed:.3f}s")


def tokens_per_second(n_tokens: int, elapsed_s: float) -> float:
    """
    Compute token throughput.

    Args:
        n_tokens:  Number of tokens processed.
        elapsed_s: Wall-clock time in seconds.

    Returns:
        Tokens per second as a float.
    """
    return n_tokens / max(elapsed_s, 1e-9)
