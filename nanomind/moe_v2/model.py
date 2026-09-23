"""
nanomind/moe_v2/model.py — Sparse MoE Transformer.

Interleaves dense attention layers with sparse MoE FFN layers.
Typical pattern (Mixtral, Switch):
  - Every layer: MoE FFN (replaces dense FFN)
  - Some architectures: alternate dense and MoE layers

Architecture:
  Token Embedding → N × (Attention + MoE FFN) → LM Head

Total params scale with n_experts × d_ff (width),
but active params per token scale with top_k × d_ff.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.moe_v2.config import MoEConfig
from nanomind.moe_v2.layer import MoELayer


class MoETransformerBlock(nn.Module):
    """
    Transformer block with MoE FFN.

    Args:
        d_model:  Model dimension.
        n_heads:  Attention heads.
        moe_cfg:  MoE configuration.
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        moe_cfg:  MoEConfig,
    ) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2  = nn.LayerNorm(d_model)
        self.moe  = MoELayer(moe_cfg)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        h, _   = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                            need_weights=False)
        x      = x + h
        # MoE FFN
        moe_out, aux = self.moe(self.ln2(x))
        x      = x + moe_out
        return x, aux


class SparseMoETransformer(nn.Module):
    """
    Full Sparse MoE Transformer language model.

    Args:
        vocab_size: Vocabulary size.
        d_model:    Model dimension.
        n_layers:   Number of transformer blocks.
        n_heads:    Attention heads.
        max_seq:    Max sequence length.
        moe_cfg:    MoE configuration.

    Example::

        cfg   = MoEConfig(n_experts=8, top_k=2, d_model=128, d_ff=512)
        model = SparseMoETransformer(vocab_size=1000, d_model=128,
                                      n_layers=4, n_heads=4, moe_cfg=cfg)
        logits, loss, aux = model(input_ids, targets)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int,
        n_layers:   int,
        n_heads:    int,
        max_seq:    int,
        moe_cfg:    MoEConfig,
    ) -> None:
        super().__init__()
        self.tok_emb  = nn.Embedding(vocab_size, d_model)
        self.pos_emb  = nn.Embedding(max_seq, d_model)
        self.blocks   = nn.ModuleList([
            MoETransformerBlock(d_model, n_heads, moe_cfg)
            for _ in range(n_layers)
        ])
        self.ln_f     = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq  = max_seq

    def forward(
        self,
        input_ids: torch.Tensor,
        targets:   torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: ``(B, T)`` token IDs.
            targets:   ``(B, T)`` target token IDs (for loss).

        Returns:
            ``(logits, loss, aux_loss)``
            where aux_loss is the sum of all MoE load-balance losses.
        """
        B, T    = input_ids.shape
        T       = min(T, self.max_seq)
        x       = self.tok_emb(input_ids[:, :T])
        x       = x + self.pos_emb(torch.arange(T))

        total_aux = torch.tensor(0.0)
        for block in self.blocks:
            x, aux  = block(x)
            total_aux = total_aux + aux

        x       = self.ln_f(x)
        logits  = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets[:, :T].contiguous().view(-1),
            )

        return logits, loss, total_aux

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def n_active_params(self) -> int:
        """Active parameters per forward pass (top_k experts only)."""
        moe = self.blocks[0].moe
        cfg = moe.cfg
        per_block = (
            cfg.top_k * moe.experts.n_params_per_expert
            + sum(p.numel() for p in self.blocks[0].attn.parameters())
        )
        return self.tok_emb.weight.numel() + len(self.blocks) * per_block
