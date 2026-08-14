"""
nanomind/model/model.py — The NanoMind GPT-style language model.

Architecture overview:
    Input IDs  (B, T)
        |
    Token Embedding   (vocab_size -> d_model)
    + Positional Emb  (block_size -> d_model)
        |
    [TransformerBlock] x N
        |
    Final LayerNorm
        |
    LM Head  (d_model -> vocab_size)   [weights tied to Token Embedding]
        |
    Logits  (B, T, vocab_size)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.model.config import ModelConfig
from nanomind.blocks import TransformerBlock, get_norm


class NanoMind(nn.Module):
    """
    NanoMind — A GPT-style causal language model.

    Args:
        cfg: :class:`~nanomind.model.ModelConfig` instance.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Token + position embeddings
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.block_size, cfg.d_model)
        self.emb_drop  = nn.Dropout(cfg.dropout)

        # Stack of N transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                block_size=cfg.block_size,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                norm_type=cfg.norm_type,
                activation=cfg.activation,
                norm_placement=cfg.norm_placement,
            )
            for _ in range(cfg.n_layers)
        ])

        # Final norm + LM head — to be filled in next commits
        self.final_norm: nn.Module | None = None
        self.lm_head:    nn.Linear | None = None
