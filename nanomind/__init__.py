"""
NanoMind - A small GPT-style transformer LLM built from scratch in PyTorch.
"""

__version__ = "0.1.0"
__author__ = "ananthan-bot"

from nanomind.utils import (
    get_logger,
    get_device,
    set_seed,
    fmt_number,
    fmt_time,
)

__all__ = [
    "__version__",
    "get_logger",
    "get_device",
    "set_seed",
    "fmt_number",
    "fmt_time",
]
