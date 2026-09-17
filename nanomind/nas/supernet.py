"""
nanomind/nas/supernet.py — Weight-sharing supernet for one-shot NAS.

A Supernet trains a single large model whose sub-networks
share weights. Any subnet can be extracted and evaluated without
retraining from scratch.

Architecture:
  Supernet has max layers, max d_model, max n_heads.
  Each forward pass randomly samples a subnet config and
  only uses a subset of the weights.

This is the core idea behind:
  - Once-for-All (Cai et al., 2019)
  - Single Path One Shot (Guo et al., 2020)
  - SPOS (Guo et al., 2020)

Training: train the supernet with full max config.
Evaluation: sample subnets and evaluate with inherited weights.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.nas.search_space import ArchConfig, SearchSpace


class SupernetBlock(nn.Module):
    """
    A transformer block with configurable active width.

    Weights are allocated for the max d_model, but during
    forward pass only the first ``active_d`` columns are used.

    Args:
        max_d:   Maximum d_model (supernet width).
        max_h:   Maximum n_heads.
        max_ffn: Maximum FFN expansion ratio.
    """

    def __init__(self, max_d: int, max_h: int, max_ffn: int) -> None:
        super().__init__()
        self.max_d   = max_d
        self.max_h   = max_h
        self.max_ffn = max_ffn
        d_ff         = max_d * max_ffn
        self.q  = nn.Linear(max_d, max_d, bias=False)
        self.k  = nn.Linear(max_d, max_d, bias=False)
        self.v  = nn.Linear(max_d, max_d, bias=False)
        self.o  = nn.Linear(max_d, max_d, bias=False)
        self.ff1 = nn.Linear(max_d, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, max_d, bias=False)
        self.ln1 = nn.LayerNorm(max_d)
        self.ln2 = nn.LayerNorm(max_d)

    def forward(
        self,
        x:        torch.Tensor,
        active_d: int,
        active_h: int,
        active_ffn: int,
    ) -> torch.Tensor:
        """Forward using only active_d dimensions (subnet mode)."""
        B, T, _ = x.shape
        d_h     = active_d // max(active_h, 1)

        # Slice weights to active dimensions
        q = F.linear(x[:, :, :active_d], self.q.weight[:active_d, :active_d])
        k = F.linear(x[:, :, :active_d], self.k.weight[:active_d, :active_d])
        v = F.linear(x[:, :, :active_d], self.v.weight[:active_d, :active_d])

        # Reshape for MHA
        q = q.view(B, T, active_h, d_h).transpose(1, 2)
        k = k.view(B, T, active_h, d_h).transpose(1, 2)
        v = v.view(B, T, active_h, d_h).transpose(1, 2)

        scale = d_h ** -0.5
        attn  = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        ctx   = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, active_d)
        ctx   = F.linear(ctx, self.o.weight[:active_d, :active_d])

        x     = x[:, :, :active_d] + ctx
        # FFN
        d_ff  = active_d * active_ffn
        h     = F.gelu(F.linear(self.ln2(x), self.ff1.weight[:d_ff, :active_d]))
        h     = F.linear(h, self.ff2.weight[:active_d, :d_ff])
        return x + h


class Supernet(nn.Module):
    """
    Weight-sharing Supernet for one-shot NAS.

    Args:
        space:      Search space defining max dimensions.
        vocab_size: Vocabulary size.

    Example::

        supernet = Supernet(SearchSpace(), vocab_size=64)
        cfg      = ArchConfig(d_model=64, n_layers=2, n_heads=4)
        logits, loss = supernet(x, y, subnet_cfg=cfg)
    """

    def __init__(self, space: SearchSpace, vocab_size: int = 64) -> None:
        super().__init__()
        self.space     = space
        self.max_d     = max(space.d_model_choices)
        self.max_l     = max(space.n_layers_choices)
        self.max_h     = max(space.n_heads_choices)
        self.max_ffn   = max(space.ffn_ratio_choices)
        self.max_T     = 32
        self.tok       = nn.Embedding(vocab_size, self.max_d)
        self.pos       = nn.Embedding(self.max_T, self.max_d)
        self.blocks    = nn.ModuleList([
            SupernetBlock(self.max_d, self.max_h, self.max_ffn)
            for _ in range(self.max_l)
        ])
        self.ln        = nn.LayerNorm(self.max_d)
        self.lm        = nn.Linear(self.max_d, vocab_size, bias=False)

    def forward(
        self,
        x:          torch.Tensor,
        t:          torch.Tensor | None = None,
        subnet_cfg: ArchConfig | None   = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass with optional subnet configuration.

        Args:
            x:          ``(B, T)`` token IDs.
            t:          ``(B, T)`` target IDs for loss.
            subnet_cfg: If given, use subnet dimensions; else use max.

        Returns:
            ``(logits, loss)``
        """
        cfg      = subnet_cfg or ArchConfig(
            d_model=self.max_d, n_layers=self.max_l,
            n_heads=self.max_h, ffn_ratio=self.max_ffn,
        )
        B, S     = x.shape
        S        = min(S, self.max_T)
        x        = x[:, :S]
        h        = self.tok(x)[:, :, :cfg.d_model]
        h        = h + self.pos(torch.arange(S))[:, :cfg.d_model]

        for i, block in enumerate(self.blocks[:cfg.n_layers]):
            h    = block(h, cfg.d_model, cfg.n_heads, cfg.ffn_ratio)

        h        = self.ln(h[:, :, :cfg.d_model] if False else h)
        # Slice LM head to active d_model
        logits   = F.linear(h[:, :, :cfg.d_model],
                             self.lm.weight[:, :cfg.d_model])
        loss     = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                    t.view(-1)) if t is not None else None
        return logits, loss

    def sample_subnet(self) -> ArchConfig:
        """Sample a random valid subnet configuration."""
        return self.space.random_sample()
