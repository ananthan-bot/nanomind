"""
nanomind/cache/model.py — NanoMind transformer with KV cache inference support.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.model.config import ModelConfig
from nanomind.cache.block import CachedTransformerBlock
from nanomind.cache.manager import KVCacheManager
from nanomind.cache.config import KVCacheConfig
from nanomind.norm.factory import get_norm
from nanomind.utils.logger import get_logger

log = get_logger("cache.model")


class NanoMindCached(nn.Module):
    """
    NanoMind transformer with integrated KV cache for fast inference.

    During prefill (processing the prompt), pass all tokens at once.
    During decode (generating new tokens), pass one token at a time
    while the KV cache stores all prior K/V tensors.

    Args:
        model_cfg: Standard model configuration.
        cache_cfg: KV cache configuration.

    Example::

        model = NanoMindCached(model_cfg, cache_cfg)
        cache = model.new_cache()

        # Prefill
        logits = model.prefill(prompt_ids, cache)

        # Decode loop
        for _ in range(max_new_tokens):
            next_token = sample(logits[:, -1])
            logits     = model.decode_step(next_token.unsqueeze(1), cache)
    """

    def __init__(self, model_cfg: ModelConfig, cache_cfg: KVCacheConfig) -> None:
        super().__init__()
        self.cfg       = model_cfg
        self.cache_cfg = cache_cfg

        self.tok_emb  = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        self.pos_emb  = nn.Embedding(model_cfg.block_size, model_cfg.d_model)
        self.drop     = nn.Dropout(model_cfg.dropout)
        self.blocks   = nn.ModuleList([
            CachedTransformerBlock(
                d_model=model_cfg.d_model,
                n_heads=model_cfg.n_heads,
                dropout=model_cfg.dropout,
            )
            for _ in range(model_cfg.n_layers)
        ])
        self.norm     = get_norm("layernorm", model_cfg.d_model)
        self.lm_head  = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)

        self._init_weights()
        n = sum(p.numel() for p in self.parameters())
        log.info(f"NanoMindCached: {n:,} params, cache={cache_cfg.cache_size_mb:.1f}MB")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def new_cache(self) -> KVCacheManager:
        """Create a fresh KVCacheManager for this model."""
        return KVCacheManager(self.cache_cfg)

    def _forward(
        self,
        idx:      torch.Tensor,
        cache:    KVCacheManager | None = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        B, T = idx.shape
        tok  = self.tok_emb(idx)
        pos  = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        x    = self.drop(tok + self.pos_emb(pos))

        for i, block in enumerate(self.blocks):
            kv = cache.get(i) if cache is not None else None
            x  = block(x, kv)

        x      = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def prefill(
        self,
        prompt_ids: torch.Tensor,
        cache:      KVCacheManager,
    ) -> torch.Tensor:
        """
        Process the prompt and populate the KV cache.

        Args:
            prompt_ids: ``(B, T_prompt)`` token IDs.
            cache:      Fresh KVCacheManager (will be filled).

        Returns:
            Logits ``(B, T_prompt, vocab_size)``.
        """
        cache.reset()
        return self._forward(prompt_ids, cache, pos_offset=0)

    def decode_step(
        self,
        token_ids: torch.Tensor,
        cache:     KVCacheManager,
    ) -> torch.Tensor:
        """
        Single decode step: attend over cached K/V, produce next logits.

        Args:
            token_ids: ``(B, 1)`` last generated token.
            cache:     Populated KVCacheManager.

        Returns:
            Logits ``(B, 1, vocab_size)``.
        """
        pos = cache.current_len
        return self._forward(token_ids, cache, pos_offset=pos)

    def forward(
        self,
        idx:     torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Standard forward (no cache) — for training compatibility."""
        logits = self._forward(idx)
        if targets is not None:
            import torch.nn.functional as F
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
            return logits, loss
        return logits, None
