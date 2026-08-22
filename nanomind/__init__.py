"""
NanoMind — A GPT-style Language Model built layer by layer.

Quick start::

    from nanomind import NanoMind, ModelConfig

    cfg   = ModelConfig(vocab_size=256, d_model=128, n_layers=4, n_heads=4)
    model = NanoMind(cfg)
    logits, loss = model(idx, targets)

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__  = "NanoMind Contributors"
__license__ = "MIT"

from nanomind.model import NanoMind, ModelConfig
from nanomind.config import NanoMindConfig

__all__ = [
    "NanoMind",
    "ModelConfig",
    "NanoMindConfig",
    "__version__",
]
