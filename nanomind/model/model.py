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
        # Learned pos embedding used only when pos_type='learned'
        self.pos_emb = (
            nn.Embedding(cfg.block_size, cfg.d_model)
            if cfg.pos_type == "learned"
            else None
        )
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
                pos_type=cfg.pos_type,
            )
            for _ in range(cfg.n_layers)
        ])

        # Final normalization before the language model head
        self.final_norm = get_norm(cfg.norm_type, cfg.d_model)

        # LM head: projects d_model -> vocab_size
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Initialize weights using GPT-2 style normal initialization
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layers) for stability
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

        # Weight tying: share token embedding matrix with LM head
        # This reduces parameters and often improves performance.
        if cfg.weight_tying:
            self.lm_head.weight = self.token_emb.weight

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 style weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Run a forward pass through NanoMind.

        Args:
            idx:     Input token IDs ``(B, T)``
            targets: Target token IDs ``(B, T)`` for loss computation.
                     If None, only logits are returned.

        Returns:
            Tuple of:
            - ``logits``: ``(B, T, vocab_size)``
            - ``loss``:   Cross-entropy loss scalar, or None if no targets.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"Sequence length {T} exceeds block_size {self.cfg.block_size}"
        )

        # Token + positional embeddings
        tok = self.token_emb(idx)   # (B, T, d_model)
        if self.pos_emb is not None:
            pos = self.pos_emb(torch.arange(T, device=idx.device))
            x   = self.emb_drop(tok + pos)
        else:
            x = self.emb_drop(tok)   # RoPE/ALiBi: no explicit pos emb

        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x)

        # Final norm + LM head
        x      = self.final_norm(x)
        logits = self.lm_head(x)                                      # (B, T, vocab_size)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
            )
        return logits, loss

    # ── Utilities ─────────────────────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.

        Args:
            trainable_only: If True, count only trainable parameters
                            (requires_grad=True). Default: True.

        Returns:
            Total parameter count as an integer.
        """
        params = (
            p for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )
        return sum(p.numel() for p in params)

    def __repr__(self) -> str:
        from nanomind.utils.format import fmt_number
        n = self.num_parameters()
        return (
            f"NanoMind("
            f"vocab={self.cfg.vocab_size}, "
            f"d_model={self.cfg.d_model}, "
            f"layers={self.cfg.n_layers}, "
            f"heads={self.cfg.n_heads}, "
            f"params={fmt_number(n)})"
        )

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """
        Autoregressively generate tokens appended to ``idx``.

        Args:
            idx:            Seed token IDs ``(B, T)``
            max_new_tokens: Number of new tokens to generate.
            temperature:    Softmax temperature. < 1 = sharper, > 1 = more random.
            top_k:          If set, only sample from the top-k logits.
            top_p:          If set, apply nucleus (top-p) sampling.

        Returns:
            Token IDs ``(B, T + max_new_tokens)``
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)
        return idx
