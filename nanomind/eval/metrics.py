"""
nanomind/eval/metrics.py — Language model evaluation metrics.

Implements the standard LM evaluation metrics:
- Perplexity (PPL): exp(cross-entropy loss) — lower is better
- Bits per character (BPC): cross-entropy in bits — lower is better
- Token accuracy: fraction of correct next-token predictions
- Top-k accuracy: correct token in top-k predictions
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def perplexity(loss: float) -> float:
    """
    Compute perplexity from a cross-entropy loss value.

    PPL = exp(H)  where H is the average cross-entropy in nats.

    Lower perplexity = better language model.
    A uniform model over V tokens has PPL = V.

    Args:
        loss: Average cross-entropy loss (in nats).

    Returns:
        Perplexity as a float.
    """
    return math.exp(min(loss, 20.0))   # cap to avoid inf for very bad models


def bits_per_character(loss: float) -> float:
    """
    Compute bits-per-character (BPC) from a cross-entropy loss.

    BPC = H / log(2)   (converts nats to bits)

    Commonly used for character-level language models.
    A good model achieves BPC ≈ 1.0-1.5 on English text.

    Args:
        loss: Average cross-entropy loss (in nats).

    Returns:
        BPC as a float.
    """
    return loss / math.log(2)


def token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute the fraction of tokens where the argmax prediction is correct.

    Args:
        logits:  Model output ``(B, T, vocab_size)`` or ``(N, vocab_size)``
        targets: Ground-truth token IDs ``(B, T)`` or ``(N,)``

    Returns:
        Accuracy as a float in [0, 1].
    """
    if logits.dim() == 3:
        logits  = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
    preds   = logits.argmax(dim=-1)
    correct = (preds == targets).float()
    return correct.mean().item()


def top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Compute the fraction of tokens where the correct token is in the top-K predictions.

    Args:
        logits:  Model output ``(B, T, vocab_size)`` or ``(N, vocab_size)``
        targets: Ground-truth token IDs ``(B, T)`` or ``(N,)``
        k:       Number of top predictions to consider.

    Returns:
        Top-K accuracy as a float in [0, 1].
    """
    if logits.dim() == 3:
        logits  = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
    _, top_preds = logits.topk(k, dim=-1)
    correct = top_preds.eq(targets.unsqueeze(-1)).any(dim=-1).float()
    return correct.mean().item()


def cross_entropy_on_batch(
    model: "torch.nn.Module",
    x: torch.Tensor,
    y: torch.Tensor,
    device: "torch.device | None" = None,
) -> float:
    """
    Compute cross-entropy loss for a single batch without gradients.

    Args:
        model:  The NanoMind model.
        x:      Input token IDs ``(B, T)``
        y:      Target token IDs ``(B, T)``
        device: Device to move tensors to.

    Returns:
        Scalar loss as a float.
    """
    import torch
    model.eval()
    with torch.no_grad():
        if device is not None:
            x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
    return loss.item()
