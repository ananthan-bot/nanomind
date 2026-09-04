"""
nanomind/flash/model.py — NanoMind transformer with Flash Attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.model.config import ModelConfig
from nanomind.flash.block import FlashTransformerBlock
from nanomind.flash.config import FlashConfig
from nanomind.norm.factory import get_norm
from nanomind.utils.logger import get_logger

log = get_logger("flash.model")


class NanoMindFlash(nn.Module):
    """
    NanoMind transformer using Flash Attention in every block.

    Drop-in replacement for the standard NanoMind model, using
    :class:`FlashTransformerBlock` with RMSNorm and SwiGLU FFN.

    Args:
        model_cfg:  Standard model configuration.
        flash_cfg:  Flash Attention configuration.

    Example::

        model = NanoMindFlash(
            ModelConfig(vocab_size=32000, d_model=512, n_layers=8, n_heads=8),
            FlashConfig(causal=True, use_torch_sdpa=True),
        )
        logits, loss = model(input_ids, targets)
    """

    def __init__(
        self,
        model_cfg:  ModelConfig,
        flash_cfg:  FlashConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg       = model_cfg
        self.flash_cfg = flash_cfg or FlashConfig()

        self.tok_emb  = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        self.pos_emb  = nn.Embedding(model_cfg.block_size, model_cfg.d_model)
        self.drop     = nn.Dropout(model_cfg.dropout)
        self.blocks   = nn.ModuleList([
            FlashTransformerBlock(
                d_model=model_cfg.d_model,
                n_heads=model_cfg.n_heads,
                flash_cfg=self.flash_cfg,
                dropout=model_cfg.dropout,
            )
            for _ in range(model_cfg.n_layers)
        ])
        self.norm     = get_norm("rmsnorm", model_cfg.d_model)
        self.lm_head  = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)

        self._init_weights()
        n = sum(p.numel() for p in self.parameters())
        backend = "torch_sdpa" if self.flash_cfg.use_torch_sdpa else "tiled"
        log.info(f"NanoMindFlash: {n:,} params | backend={backend}")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        idx:     torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        tok  = self.tok_emb(idx)
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(tok + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x      = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
