"""
nanomind/speculative/draft.py — Draft token generation from the small model.

The draft model generates K candidate tokens autoregressively.
Each draft step also records the probability the draft model assigned
to the chosen token, which is needed for the rejection sampling step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def generate_draft(
    draft_model: nn.Module,
    idx: torch.Tensor,
    n_draft: int,
    temperature: float = 1.0,
    top_k: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate ``n_draft`` candidate tokens from the draft (small) model.

    Args:
        draft_model: Small, fast language model.
        idx:         Current token sequence ``(1, T)``.
        n_draft:     Number of tokens to speculatively generate.
        temperature: Sampling temperature.
        top_k:       Top-K filter (0 = no filtering).

    Returns:
        Tuple of:
        - ``draft_ids``  : Generated token IDs ``(n_draft,)``
        - ``draft_probs``: Draft model probability for each chosen token ``(n_draft,)``
    """
    draft_model.eval()
    block_size = getattr(draft_model, "cfg", None) and draft_model.cfg.block_size or 512

    draft_ids:   list[int]   = []
    draft_probs: list[float] = []

    current = idx.clone()

    for _ in range(n_draft):
        ctx     = current[:, -block_size:]
        logits, _ = draft_model(ctx)
        logits  = logits[0, -1, :] / max(temperature, 1e-8)

        if top_k > 0:
            from nanomind.generate.logit_processors import apply_top_k
            logits = apply_top_k(logits, top_k)

        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        p_draft = probs[next_id].item()

        draft_ids.append(next_id.item())
        draft_probs.append(p_draft)
        current = torch.cat([current, next_id.unsqueeze(0)], dim=1)

    return (
        torch.tensor(draft_ids,  dtype=torch.long,  device=idx.device),
        torch.tensor(draft_probs, dtype=torch.float, device=idx.device),
    )
