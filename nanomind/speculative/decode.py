"""
nanomind/speculative/decode.py — Main speculative decoding loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.speculative.config import SpeculativeConfig
from nanomind.speculative.draft import generate_draft
from nanomind.speculative.verify import verify_draft
from nanomind.speculative.sampling import rejection_sample


@torch.no_grad()
def speculative_decode(
    target_model: nn.Module,
    draft_model:  nn.Module,
    idx:          torch.Tensor,
    cfg:          SpeculativeConfig | None = None,
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Full speculative decoding loop.

    Generates tokens using the draft model for proposals and the
    target model for verification, accepting tokens according to the
    rejection sampling algorithm.

    Args:
        target_model:  Large, accurate target model.
        draft_model:   Small, fast draft model.
        idx:           Initial token sequence ``(1, T)``
        cfg:           Speculative decoding configuration.
        eos_token_id:  Stop generation when this token is produced.

    Returns:
        Tuple of:
        - Generated token sequence ``(1, T + generated)``
        - Stats dict with ``n_tokens``, ``n_draft_calls``, ``acceptance_rate``
    """
    cfg = cfg or SpeculativeConfig()
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    target_model.eval()
    draft_model.eval()

    generated       = 0
    total_draft     = 0
    total_accepted  = 0
    n_draft_calls   = 0

    current = idx.clone()

    while generated < cfg.max_new_tokens:
        # How many draft tokens to generate this step
        remaining  = cfg.max_new_tokens - generated
        n_this     = min(cfg.n_draft, remaining)

        # 1. Draft: generate n_this candidate tokens
        draft_ids, draft_probs = generate_draft(
            draft_model, current,
            n_draft=n_this,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
        )
        n_draft_calls += 1
        total_draft   += n_this

        # 2. Verify: target model scores the draft tokens
        target_probs_at_draft, target_logits = verify_draft(
            target_model, current, draft_ids,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )

        # 3. Reject/accept via rejection sampling
        accepted_tokens, n_accepted = rejection_sample(
            draft_ids, draft_probs,
            target_probs_at_draft, target_logits,
            temperature=cfg.temperature,
        )

        total_accepted += n_accepted

        # Append accepted tokens to sequence
        for tok in accepted_tokens.tolist():
            current = torch.cat(
                [current, torch.tensor([[tok]], device=current.device)], dim=1
            )
            generated += 1

            if eos_token_id is not None and tok == eos_token_id:
                break
            if generated >= cfg.max_new_tokens:
                break

        if eos_token_id is not None and current[0, -1].item() == eos_token_id:
            break

    acceptance_rate = total_accepted / max(total_draft, 1)
    stats = {
        "n_tokens":       generated,
        "n_draft_calls":  n_draft_calls,
        "total_draft":    total_draft,
        "total_accepted": total_accepted,
        "acceptance_rate": acceptance_rate,
    }
    return current, stats
