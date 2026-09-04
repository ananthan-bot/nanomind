"""NanoMind Flash Attention sub-package.

Flash Attention avoids materialising the O(N²) attention weight matrix
by processing K/V in tiles using online softmax accumulation.

  Standard attention memory : O(N²) — the attention matrix
  Flash Attention memory    : O(N)  — only tile buffers in SRAM

Primary exports:
    - :class:`NanoMindFlash`        — full transformer with Flash Attention
    - :class:`FlashAttention`       — drop-in SDPA replacement (torch.sdpa / tiled)
    - :class:`FlashTransformerBlock`— RMSNorm + FlashAttention + SwiGLU block
    - :class:`FlashConfig`          — block_q, block_kv, causal, use_torch_sdpa
    - :class:`OnlineSoftmaxState`   — streaming max/sum accumulator
    - :func:`tiled_flash_attention` — pure-PyTorch reference implementation
    - :func:`standard_attention_memory`  — theoretical standard SDPA memory
    - :func:`flash_attention_memory`     — theoretical Flash Attention memory
    - :func:`memory_comparison_report`   — print N×N vs tile memory analysis
"""

from nanomind.flash.config import FlashConfig
from nanomind.flash.online_softmax import OnlineSoftmaxState
from nanomind.flash.tiled import tiled_flash_attention
from nanomind.flash.memory import (
    standard_attention_memory,
    flash_attention_memory,
    memory_comparison_report,
)
from nanomind.flash.module import FlashAttention
from nanomind.flash.block import FlashTransformerBlock
from nanomind.flash.model import NanoMindFlash

__all__ = [
    "FlashConfig",
    "OnlineSoftmaxState",
    "tiled_flash_attention",
    "standard_attention_memory",
    "flash_attention_memory",
    "memory_comparison_report",
    "FlashAttention",
    "FlashTransformerBlock",
    "NanoMindFlash",
]
