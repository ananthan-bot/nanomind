"""
nanomind/generate/beam.py — Basic beam search decoder for NanoMind.

Beam search maintains ``num_beams`` candidate sequences in parallel and
selects the globally most probable sequence, unlike greedy which only
keeps one at each step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def beam_search(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    num_beams: int = 4,
    eos_token_id: int | None = None,
    block_size: int | None = None,
) -> torch.Tensor:
    """
    Simple beam search decoder.

    Args:
        model:          The NanoMind model.
        idx:            Seed token IDs ``(1, T)`` — single sequence only.
        max_new_tokens: Maximum tokens to generate.
        num_beams:      Number of beams to maintain.
        eos_token_id:   If provided, stop beam when all beams hit EOS.
        block_size:     Maximum context length (defaults to model.cfg.block_size).

    Returns:
        Best token sequence ``(1, T + generated)``
    """
    model.eval()
    device     = idx.device
    block_size = block_size or getattr(model, "cfg", None) and model.cfg.block_size or 512

    # Initialise beams: (score, sequence)
    beams: list[tuple[float, torch.Tensor]] = [(0.0, idx.clone())]

    for _ in range(max_new_tokens):
        all_candidates: list[tuple[float, torch.Tensor]] = []

        for score, seq in beams:
            # Check EOS
            if eos_token_id is not None and seq[0, -1].item() == eos_token_id:
                all_candidates.append((score, seq))
                continue

            ctx     = seq[:, -block_size:]
            logits, _ = model(ctx)
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)

            # Expand each beam by num_beams candidates
            topk_log_probs, topk_ids = log_probs.topk(num_beams)
            for lp, tok_id in zip(topk_log_probs, topk_ids):
                new_seq = torch.cat([seq, tok_id.unsqueeze(0).unsqueeze(0)], dim=1)
                all_candidates.append((score + lp.item(), new_seq))

        # Keep top num_beams candidates by score
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:num_beams]

        # Early stop if all beams end with EOS
        if eos_token_id is not None:
            if all(b[1][0, -1].item() == eos_token_id for b in beams):
                break

    # Return the highest-scoring sequence
    best_score, best_seq = beams[0]
    return best_seq
