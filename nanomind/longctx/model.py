"""
nanomind/longctx/model.py — Long-Context Transformer LM.

Assembles a full language model using efficient attention mechanisms.
Configurable to use any combination of:
  - RoPE position encoding
  - ALiBi position bias
  - Grouped Query Attention (GQA/MQA)
  - Sliding Window Attention
  - Linear Attention
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.longctx.rope import RotaryEmbedding
from nanomind.longctx.alibi import ALiBi
from nanomind.longctx.gqa import GroupedQueryAttention
from nanomind.longctx.sliding_window import SlidingWindowAttention
from nanomind.longctx.linear_attn import LinearAttention


class LongContextConfig:
    """Configuration for a long-context transformer."""
    def __init__(
        self,
        vocab_size:   int   = 1000,
        d_model:      int   = 128,
        n_layers:     int   = 4,
        n_heads:      int   = 4,
        n_kv_heads:   int   = 4,
        max_seq:      int   = 512,
        window_size:  int   = 128,
        n_sinks:      int   = 4,
        attn_type:    str   = "gqa",    # gqa | sliding | linear
        pos_encoding: str   = "rope",   # rope | alibi | none
        dropout:      float = 0.0,
    ) -> None:
        self.vocab_size   = vocab_size
        self.d_model      = d_model
        self.n_layers     = n_layers
        self.n_heads      = n_heads
        self.n_kv_heads   = n_kv_heads
        self.max_seq      = max_seq
        self.window_size  = window_size
        self.n_sinks      = n_sinks
        self.attn_type    = attn_type
        self.pos_encoding = pos_encoding
        self.dropout      = dropout


class LongContextBlock(nn.Module):
    """Single transformer block with configurable attention."""

    def __init__(self, cfg: LongContextConfig) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.ln2  = nn.LayerNorm(cfg.d_model)

        # Attention
        if cfg.attn_type == "gqa":
            self.attn = GroupedQueryAttention(
                cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                max_seq=cfg.max_seq, use_rope=(cfg.pos_encoding == "rope")
            )
        elif cfg.attn_type == "sliding":
            self.attn = SlidingWindowAttention(
                cfg.d_model, cfg.n_heads, cfg.window_size, cfg.n_sinks
            )
        elif cfg.attn_type == "linear":
            self.attn = LinearAttention(cfg.d_model, cfg.n_heads)
        else:
            raise ValueError(f"Unknown attn_type: {cfg.attn_type!r}")

        self.attn_type = cfg.attn_type
        self.alibi = ALiBi(cfg.n_heads, cfg.max_seq) if cfg.pos_encoding == "alibi" else None

        # FFN
        d_ff = cfg.d_model * 4
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, d_ff), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        h  = self.ln1(x)
        if self.attn_type == "linear":
            h = self.attn(h)
            kv = None
        else:
            h, kv = self.attn(h)
        x = x + h
        # FFN
        x = x + self.ff(self.ln2(x))
        return x


class LongContextLM(nn.Module):
    """
    Long-Context Language Model.

    Supports GQA, sliding window, and linear attention variants.

    Args:
        cfg: :class:`LongContextConfig`.
    """

    def __init__(self, cfg: LongContextConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Position embeddings only for non-RoPE, non-ALiBi
        if cfg.pos_encoding == "none":
            self.pos_emb = nn.Embedding(cfg.max_seq, cfg.d_model)
        else:
            self.pos_emb = None
        self.blocks  = nn.ModuleList([LongContextBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets:   torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T    = input_ids.shape
        T       = min(T, self.cfg.max_seq)
        x       = self.tok_emb(input_ids[:, :T])
        if self.pos_emb is not None:
            x   = x + self.pos_emb(torch.arange(T))

        for block in self.blocks:
            x = block(x)

        x      = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets[:, :T].contiguous().view(-1),
            )
        return logits, loss

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def effective_context(self) -> int:
        """Effective context length."""
        if self.cfg.attn_type == "sliding":
            first_swa = next(
                b.attn for b in self.blocks
                if isinstance(b.attn, SlidingWindowAttention)
            )
            return first_swa.effective_context(self.cfg.n_layers)
        return self.cfg.max_seq
