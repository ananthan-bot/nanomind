"""
nanomind/utils/logger.py — Structured logging utility for NanoMind.

Provides a consistent logger with:
- Timestamped, colour-coded console output via the standard logging module
- A single get_logger() factory so every module gets the same format
"""

import logging
import sys
from typing import Optional

# ── Colour codes for terminal output ─────────────────────────────────────────
_COLOURS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET":    "\033[0m",
}

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


class _ColouredFormatter(logging.Formatter):
    """Custom formatter that adds ANSI colour codes to log level names."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        reset = _COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname}{reset}"
        return super().format(record)


def get_logger(name: str = "nanomind", level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with coloured console output.

    Args:
        name:  Logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).

    Returns:
        Configured :class:`logging.Logger` instance.

    Example::

        log = get_logger(__name__)
        log.info("Training started")
        log.warning("GPU not detected, falling back to CPU")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        _ColouredFormatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    )

    logger.addHandler(handler)
    logger.propagate = False

    return logger
