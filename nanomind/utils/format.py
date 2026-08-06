"""
nanomind/utils/format.py — Human-readable formatting helpers.
"""


def fmt_number(n: int) -> str:
    """Format a large integer with K/M/B suffix.

    Example: 1_200_000 -> "1.2M"
    """
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def fmt_time(seconds: float) -> str:
    """Format seconds as a human-readable duration.

    Example: 3661 -> "1h 01m 01s"
    """
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{seconds:.2f}s"


def fmt_loss(loss: float) -> str:
    """Format a loss value with 4 decimal places."""
    return f"{loss:.4f}"


def fmt_lr(lr: float) -> str:
    """Format a learning rate in scientific notation."""
    return f"{lr:.2e}"
