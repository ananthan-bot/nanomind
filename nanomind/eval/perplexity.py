"""
nanomind/eval/perplexity.py — Perplexity evaluation for language models.

Perplexity (PPL) is the standard intrinsic metric for language model quality:

  PPL = exp(-1/N Σ_{i=1}^{N} log p_θ(w_i | w_1 ... w_{i-1}))

Interpretation:
  PPL = 1    → perfect model (assigns probability 1 to every correct token)
  PPL = V    → random model (assigns uniform probability, V = vocab size)
  PPL ↓      → better model

Typical values:
  Character-level models: PPL 2-5 (small alphabet)
  Word-level models:      PPL 20-100 on PTB, WikiText
  GPT-2 (1.5B):          PPL ~17.48 on WikiText-103
  LLaMA 2 (70B):         PPL ~3.3 on WikiText-2

Lower perplexity = better language understanding.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_perplexity(
    model:      nn.Module,
    loader:     DataLoader,
    device:     str | torch.device = "cpu",
    max_batches: int | None = None,
) -> dict:
    """
    Compute perplexity of a language model on a dataset.

    Args:
        model:       Language model with ``forward(x, y) -> (logits, loss)``.
        loader:      DataLoader of ``(x, y)`` token batches.
        device:      Evaluation device.
        max_batches: Optionally limit to this many batches.

    Returns:
        Dict with ``perplexity``, ``nll`` (mean negative log-likelihood),
        ``n_tokens``, ``n_batches``.
    """
    model.eval()
    device   = torch.device(device)
    total_nll, n_tokens, n_batches = 0.0, 0, 0

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y  = x.to(device), y.to(device)
        _, loss = model(x, y)
        if loss is not None:
            total_nll += loss.item() * y.numel()
            n_tokens  += y.numel()
        n_batches += 1

    if n_tokens == 0:
        return {"perplexity": float("inf"), "nll": float("inf"),
                "n_tokens": 0, "n_batches": 0}

    mean_nll   = total_nll / n_tokens
    perplexity = math.exp(min(mean_nll, 100))   # cap at exp(100) to avoid overflow
    return {
        "perplexity": perplexity,
        "nll":        mean_nll,
        "n_tokens":   n_tokens,
        "n_batches":  n_batches,
        "bits_per_char": mean_nll / math.log(2),
    }


def perplexity_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute perplexity directly from logits and targets.

    Args:
        logits:  ``(N, V)`` model output logits.
        targets: ``(N,)`` target token IDs.

    Returns:
        Perplexity (scalar float).
    """
    nll = F.cross_entropy(logits, targets).item()
    return math.exp(min(nll, 100))
