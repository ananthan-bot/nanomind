"""
nanomind/utils/format.py — String formatting utilities.
"""

from __future__ import annotations


def fmt_number(n: int | float) -> str:
    """Format a large number with K/M/B suffix."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n))


def fmt_time(seconds: float) -> str:
    """Format seconds as a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def fmt_loss(loss: float) -> str:
    """Format a loss value to 4 decimal places."""
    return f"{loss:.4f}"


def fmt_lr(lr: float) -> str:
    """Format a learning rate in scientific notation."""
    return f"{lr:.2e}"


def tokens_per_second(n_tokens: int, elapsed_s: float) -> float:
    """Compute tokens/second throughput."""
    return n_tokens / max(elapsed_s, 1e-9)
