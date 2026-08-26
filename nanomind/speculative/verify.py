"""
nanomind/speculative/verify.py — Target model verification of draft tokens.

The key insight of speculative decoding: instead of running the target model
K times sequentially, we run it ONCE on the full draft sequence and extract
the probability it assigns to each draft token in parallel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def verify_draft(
    target_model: nn.Module,
    idx: torch.Tensor,
    draft_ids: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the target model on the context + draft tokens and extract probabilities.

    The target model processes ``[context | draft_ids]`` in one forward pass.
    We extract the probability distributions at each of the K draft positions.

    Args:
        target_model: Large, accurate language model.
        idx:          Original context ``(1, T)``.
        draft_ids:    Draft token IDs to verify ``(n_draft,)``.
        temperature:  Temperature for target distribution.
        top_k:        Top-K filter on target logits.
        top_p:        Nucleus filter on target logits.

    Returns:
        Tuple of:
        - ``target_probs_at_draft`` : probability of each draft token under target ``(n_draft,)``
        - ``target_logits``         : full logit distributions at each position ``(n_draft+1, V)``
    """
    target_model.eval()
    n_draft    = draft_ids.shape[0]
    block_size = getattr(target_model, "cfg", None) and target_model.cfg.block_size or 512

    # Build full input: context + draft tokens
    draft_seq   = draft_ids.unsqueeze(0)              # (1, n_draft)
    full_input  = torch.cat([idx, draft_seq], dim=1)  # (1, T + n_draft)
    ctx         = full_input[:, -block_size:]

    logits, _ = target_model(ctx)                     # (1, T+n_draft, V)

    # Extract logits at positions that predict each draft token
    # Position i in ctx predicts token at position i+1
    T_orig = min(idx.shape[1], block_size)
    # The draft tokens start at index T_orig in ctx
    # draft token j is at position T_orig + j in full_input
    # logit predicting draft token j is at ctx position T_orig - 1 + j
    start = T_orig - 1
    draft_logits = logits[0, start:start + n_draft, :]   # (n_draft, V)

    # Apply temperature and optional filtering
    draft_logits = draft_logits / max(temperature, 1e-8)
    if top_k > 0:
        from nanomind.generate.logit_processors import apply_top_k
        draft_logits = torch.stack([apply_top_k(row, top_k) for row in draft_logits])
    if top_p > 0.0:
        from nanomind.generate.logit_processors import apply_top_p
        draft_logits = torch.stack([apply_top_p(row, top_p) for row in draft_logits])

    target_probs = F.softmax(draft_logits, dim=-1)  # (n_draft, V)

    # Probability the target assigns to each chosen draft token
    probs_at_draft = target_probs[
        torch.arange(n_draft, device=idx.device), draft_ids
    ]                                               # (n_draft,)

    # Also return the next-token logits AFTER the last draft token (bonus token)
    bonus_logits = logits[0, start + n_draft, :]   # (V,)
    all_logits   = torch.cat([draft_logits, bonus_logits.unsqueeze(0)], dim=0)  # (n_draft+1, V)

    return probs_at_draft, all_logits
