"""
nanomind/specd/medusa.py — Medusa: self-speculation with multiple LM heads.

## Medusa (Cai et al., 2024)

Standard speculative decoding needs TWO models (draft + target).
Medusa adds K extra "heads" to the target model itself:

  Token n:
    Head 0 (original): predicts token n+1 (standard LM head)
    Head 1 (Medusa):   predicts token n+2 (2-ahead)
    Head 2 (Medusa):   predicts token n+3 (3-ahead)
    ...
    Head K (Medusa):   predicts token n+K+1 (K+1 ahead)

These heads are trained with teacher forcing on the same input,
using only a fraction of the training compute.

Verification: run one forward pass, get K+1 candidate tokens,
accept greedily or via tree attention.

Benefits:
  ✓ Single model (no separate draft model needed)
  ✓ 2-3× speedup on most tasks
  ✓ Easy to add to any transformer

## Tree Attention (for candidate expansion)

Instead of K linear candidates, Medusa uses tree attention to
explore multiple hypotheses simultaneously:

         root
        / | \
       A  B  C       ← Head 1 top-3
      /|  |  |\
     D E  F  G H     ← Head 2 top-{2,1,2}

This explores more candidates with the same compute budget.

Reference:
  Cai et al. (2024) "Medusa: Simple LLM Inference Acceleration Framework
  with Multiple Decoding Heads"
  https://arxiv.org/abs/2401.10774
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MedusaHead(nn.Module):
    """
    A single Medusa lookahead head.

    A lightweight 2-layer MLP on top of the LM hidden states,
    trained to predict the token K positions ahead.

    Args:
        d_model:    LM hidden dimension.
        vocab_size: Vocabulary size.
        d_ff:       Hidden dim of Medusa MLP (default = 2 × d_model).
    """

    def __init__(self, d_model: int, vocab_size: int, d_ff: int | None = None) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 2
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, vocab_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: ``(B, T, D)`` LM hidden states.
        Returns:
            ``(B, T, V)`` logits for k-ahead token.
        """
        return self.net(hidden)


class MedusaModel(nn.Module):
    """
    LM model augmented with K Medusa speculation heads.

    Args:
        base_model:  The base LM (must return hidden states + logits).
        n_heads:     Number of Medusa heads (lookahead steps).
        d_model:     LM hidden dimension.
        vocab_size:  Vocabulary size.

    Example::

        model  = MedusaModel(base_lm, n_heads=3, d_model=128, vocab_size=1000)
        logits, medusa_logits = model(input_ids)
        # logits:         (B, T, V)   — standard next token
        # medusa_logits:  list of 3 × (B, T, V)  — 2,3,4-ahead
    """

    def __init__(
        self,
        base_model: nn.Module,
        n_heads:    int,
        d_model:    int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.base       = base_model
        self.n_heads    = n_heads
        self.medusa_heads = nn.ModuleList([
            MedusaHead(d_model, vocab_size)
            for _ in range(n_heads)
        ])
        self.vocab_size = vocab_size
        self.d_model    = d_model

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass with Medusa heads.

        Args:
            input_ids: ``(B, T)`` token IDs.

        Returns:
            ``(base_logits, medusa_logits)``
            base_logits:   ``(B, T, V)``
            medusa_logits: list of K tensors, each ``(B, T, V)``
        """
        # Get base model output
        base_out = self.base(input_ids)
        if isinstance(base_out, tuple):
            base_logits = base_out[0]
        else:
            base_logits = base_out

        # Use base logits as proxy for hidden states
        # (In production: extract hidden states from LM, not logits)
        hidden = base_logits   # (B, T, V) used as features

        medusa_logits = [head(hidden) for head in self.medusa_heads]
        return base_logits, medusa_logits

    @torch.no_grad()
    def speculate(
        self,
        input_ids:  torch.Tensor,
        top_k_each: int = 1,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generate K+1 candidate token sequences.

        Returns greedy predictions from each head.

        Args:
            input_ids:  ``(B, T)`` context.
            top_k_each: Top-K candidates per head (for tree decoding).

        Returns:
            ``(base_token, medusa_tokens)``
            base_token:    ``(B, 1)`` base model's next token
            medusa_tokens: list of K tensors ``(B, top_k_each)`` per head
        """
        base_logits, medusa_logits = self.forward(input_ids)
        base_token    = base_logits[:, -1, :].topk(top_k_each, dim=-1).indices
        medusa_tokens = [ml[:, -1, :].topk(top_k_each, dim=-1).indices
                         for ml in medusa_logits]
        return base_token, medusa_tokens

    def medusa_loss(
        self,
        input_ids: torch.Tensor,
        labels:    torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Compute base + Medusa training losses.

        For head k, the label is the token k+2 positions ahead
        (shifted by k+1 from the base label).

        Args:
            input_ids: ``(B, T)`` tokens.
            labels:    ``(B, T)`` next-token labels.

        Returns:
            ``(base_loss, medusa_losses)``
        """
        base_logits, medusa_logits = self.forward(input_ids)
        B, T, V = base_logits.shape

        # Base loss: standard next-token prediction
        base_loss = F.cross_entropy(
            base_logits[:, :-1].reshape(-1, V),
            labels[:, 1:].reshape(-1),
        )

        # Medusa losses: k+2 ahead
        medusa_losses = []
        for k, ml in enumerate(medusa_logits):
            shift = k + 2
            if shift >= T:
                medusa_losses.append(torch.tensor(0.0))
                continue
            loss = F.cross_entropy(
                ml[:, :-shift].reshape(-1, V),
                labels[:, shift:].reshape(-1),
            )
            medusa_losses.append(loss)

        return base_loss, medusa_losses

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def n_medusa_params(self) -> int:
        return sum(p.numel() for p in self.medusa_heads.parameters())
