"""NanoMind transformer blocks sub-package.

Core components:
    - :class:`TransformerBlock`  — Pre/Post-Norm transformer block
    - :class:`FeedForward`       — Position-wise FFN (GELU or SwiGLU)
    - :class:`LayerNorm`         — Standard layer normalization
    - :class:`RMSNorm`           — RMS normalization (LLaMA-style)
    - :class:`BlockConfig`       — Block configuration dataclass
    - :func:`get_norm`           — Norm factory by name
    - :func:`get_ffn`            — FFN factory by activation name
    - :func:`block_from_config`  — Block factory from config
"""

from nanomind.blocks.norms import LayerNorm, RMSNorm, get_norm, list_norms
from nanomind.blocks.feedforward import FeedForward, get_ffn
from nanomind.blocks.block import TransformerBlock, block_from_config
from nanomind.blocks.config import BlockConfig

__all__ = [
    "LayerNorm",
    "RMSNorm",
    "get_norm",
    "list_norms",
    "FeedForward",
    "get_ffn",
    "TransformerBlock",
    "block_from_config",
    "BlockConfig",
]
