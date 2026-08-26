"""
nanomind/speculative/sampling.py — Speculative decoding rejection sampling.

Implements the modified rejection sampling algorithm that guarantees the
output distribution matches the target model's distribution exactly,
regardless of which draft tokens are accepted or rejected.

Algorithm (Leviathan et al. 2022):
  For each draft token t_i with draft prob q_i and target prob p_i:
    - Accept with probability min(1, p_i / q_i)
    - If rejected: sample a correction token from max(0, p - q) / Z

This ensures the *expected* output matches sampling from the target model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rejection_sample(
    draft_ids:     torch.Tensor,
    draft_probs:   torch.Tensor,
    target_probs_at_draft: torch.Tensor,
    target_logits: torch.Tensor,
    temperature:   float = 1.0,
) -> tuple[torch.Tensor, int]:
    """
    Apply speculative decoding rejection sampling.

    Args:
        draft_ids:             Draft token IDs ``(n_draft,)``
        draft_probs:           Draft model probability for each token ``(n_draft,)``
        target_probs_at_draft: Target model probability for each draft token ``(n_draft,)``
        target_logits:         Full target logit distributions ``(n_draft+1, V)``
        temperature:           Target sampling temperature.

    Returns:
        Tuple of:
        - ``accepted_tokens``: Accepted token IDs (length 1 to n_draft+1)
        - ``n_accepted``:      Number of draft tokens accepted (0 to n_draft)
    """
    n_draft  = draft_ids.shape[0]
    accepted = []

    for i in range(n_draft):
        q_i = draft_probs[i].clamp(min=1e-8)
        p_i = target_probs_at_draft[i].clamp(min=1e-8)

        # Accept probability: min(1, p/q)
        accept_prob = torch.minimum(torch.ones(1, device=q_i.device), p_i / q_i)
        u = torch.rand(1, device=accept_prob.device)

        if u.item() < accept_prob.item():
            accepted.append(draft_ids[i].item())
        else:
            # Rejected: sample correction token from max(0, p - q)
            target_full = F.softmax(target_logits[i] / max(temperature, 1e-8), dim=-1)
            draft_full  = torch.zeros_like(target_full)
            draft_full[draft_ids[i]] = q_i

            corrected = (target_full - draft_full).clamp(min=0.0)
            z = corrected.sum()
            if z > 1e-8:
                corrected /= z
                correction_tok = torch.multinomial(corrected, num_samples=1)
            else:
                correction_tok = target_full.argmax().unsqueeze(0)
            accepted.append(correction_tok.item())
            # Stop at first rejection
            return torch.tensor(accepted, dtype=torch.long), len(accepted) - 1

    # All draft tokens accepted — sample one bonus token from target
    bonus_probs = F.softmax(target_logits[-1] / max(temperature, 1e-8), dim=-1)
    bonus_tok   = torch.multinomial(bonus_probs, num_samples=1)
    accepted.append(bonus_tok.item())
    return torch.tensor(accepted, dtype=torch.long), n_draft
