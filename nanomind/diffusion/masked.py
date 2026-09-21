"""
nanomind/diffusion/masked.py — Masked Diffusion Language Model (D3PM / MDM).

Masked diffusion treats the forward process as progressive masking:
  q(x_t | x_{t-1}): independently mask each token with probability β_t
  q(x_T | x_0):     all tokens are masked with high probability

The model learns to recover masked tokens:
  p_θ(x_0 | x_t) ≈ BERT-style MLM objective

This is closely related to BERT (masked language modelling) but:
  - BERT masks 15% randomly, fixed
  - MDM masks at varying rates controlled by the schedule
  - MDM has a principled generative process

At generation time:
  1. Start with all-MASK tokens
  2. Iteratively unmask tokens using learned p_θ(x_0 | x_t)

Used in: MDLM (Sahoo et al., 2024), SEDD (Lou et al., 2024).

Reference:
  Austin et al. (2021) "Structured Denoising Diffusion" D3PM
  https://arxiv.org/abs/2107.03006

  Sahoo et al. (2024) MDLM: https://arxiv.org/abs/2406.07524
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


MASK_ID = 1   # Token ID used as [MASK]


class MaskedDiffusionLM(nn.Module):
    """
    Masked Diffusion Language Model.

    Forward: mask tokens at rate controlled by schedule.
    Reverse: predict original tokens from masked input (MLM).

    Args:
        vocab_size: Vocabulary size (including MASK_ID=1).
        d_model:    Model dimension.
        n_layers:   Transformer layers.
        n_heads:    Attention heads.
        max_seq:    Max sequence length.

    Example::

        mdlm = MaskedDiffusionLM(vocab_size=100, d_model=64)
        loss = mdlm.loss(token_ids, t=50, T=100)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int = 64,
        n_layers:   int = 3,
        n_heads:    int = 4,
        max_seq:    int = 64,
    ) -> None:
        super().__init__()
        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.tok         = nn.Embedding(vocab_size, d_model)
        self.pos         = nn.Embedding(max_seq, d_model)
        self.t_emb       = nn.Embedding(1001, d_model)
        self.blocks      = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.head        = nn.Linear(d_model, vocab_size)

    def _mask_rate(self, t: int, T: int) -> float:
        """Linear mask rate: 0 at t=0, 1 at t=T."""
        return t / max(T, 1)

    def forward_mask(
        self,
        x:  torch.Tensor,
        t:  int,
        T:  int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: randomly mask tokens.

        Args:
            x:  ``(B, L)`` clean token IDs.
            t:  Diffusion step.
            T:  Total steps.

        Returns:
            ``(x_masked, mask)`` where mask is True at masked positions.
        """
        rate   = self._mask_rate(t, T)
        mask   = torch.rand_like(x.float()) < rate
        x_mask = x.clone()
        x_mask[mask] = MASK_ID
        return x_mask, mask

    def predict(
        self,
        x_masked: torch.Tensor,
        t:        int,
    ) -> torch.Tensor:
        """
        Predict logits for all positions.

        Args:
            x_masked: ``(B, L)`` masked token IDs.
            t:        Diffusion step.

        Returns:
            ``(B, L, V)`` logits.
        """
        B, L    = x_masked.shape
        pos     = torch.arange(L)
        ts      = torch.full((B,), min(t, 1000), dtype=torch.long)
        h       = self.tok(x_masked) + self.pos(pos) + self.t_emb(ts).unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        return self.head(h)

    def loss(
        self,
        token_ids: torch.Tensor,
        t:         int,
        T:         int = 1000,
    ) -> torch.Tensor:
        """
        Compute masked diffusion training loss.

        Only compute loss on masked positions (like BERT MLM).
        """
        x_masked, mask = self.forward_mask(token_ids, t, T)
        logits         = self.predict(x_masked, t)         # (B, L, V)

        # Only penalise masked tokens
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)

        flat_logits = logits[mask]                          # (M, V)
        flat_labels = token_ids[mask]                       # (M,)
        return F.cross_entropy(flat_logits, flat_labels)

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        seq_len:    int = 16,
        T:          int = 20,
    ) -> torch.Tensor:
        """
        Generate text via iterative unmasking.

        Start fully masked, unmask top-confidence tokens each step.
        """
        x = torch.full((batch_size, seq_len), MASK_ID, dtype=torch.long)

        for step in range(T, 0, -1):
            logits = self.predict(x, step)   # (B, L, V)
            probs  = torch.softmax(logits, dim=-1)
            tokens = probs.argmax(dim=-1)    # (B, L)

            # Only unmask: keep already unmasked, sample new ones
            is_masked     = (x == MASK_ID)
            confidence, _ = probs.max(dim=-1)   # (B, L)
            # Unmask top-confidence masked positions
            n_unmask = max(1, int(seq_len * step / T))
            for b in range(batch_size):
                masked_pos = is_masked[b].nonzero(as_tuple=True)[0]
                if len(masked_pos) == 0:
                    continue
                conf_b = confidence[b][masked_pos]
                n      = min(n_unmask, len(masked_pos))
                top_k  = conf_b.topk(n).indices
                chosen = masked_pos[top_k]
                x[b, chosen] = tokens[b, chosen]
        return x
