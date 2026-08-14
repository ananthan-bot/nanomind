"""NanoMind model sub-package.

Primary exports:
    - :class:`NanoMind`    — the full GPT-style LLM
    - :class:`ModelConfig` — architecture configuration dataclass
"""

from nanomind.model.config import ModelConfig
from nanomind.model.model import NanoMind

__all__ = ["NanoMind", "ModelConfig"]
