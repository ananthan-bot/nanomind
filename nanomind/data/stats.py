"""
nanomind/data/stats.py — Dataset statistics and diagnostics.
"""

from __future__ import annotations

import torch

from nanomind.tokenizer.base import BaseTokenizer


def dataset_stats(text: str, tokenizer: BaseTokenizer) -> dict:
    """
    Compute basic statistics about a tokenized corpus.

    Args:
        text:      Raw corpus string.
        tokenizer: A fitted tokenizer.

    Returns:
        Dictionary with keys:
        - ``num_chars``     : Total characters
        - ``num_tokens``    : Total tokens after encoding
        - ``vocab_size``    : Tokenizer vocabulary size
        - ``coverage``      : Fraction of unique chars in tokenizer vocab
        - ``compression``   : tokens / chars ratio (< 1 means compression)
    """
    ids = tokenizer.encode(text)
    unique_chars = set(text)
    covered = sum(1 for ch in unique_chars if ch in getattr(tokenizer, "_char_to_id", {}))

    return {
        "num_chars":   len(text),
        "num_tokens":  len(ids),
        "vocab_size":  tokenizer.vocab_size,
        "coverage":    covered / max(len(unique_chars), 1),
        "compression": len(ids) / max(len(text), 1),
    }


def print_stats(stats: dict) -> None:
    """Pretty-print dataset statistics."""
    print("Dataset Statistics")
    print("─" * 35)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<15}: {v:.4f}")
        else:
            print(f"  {k:<15}: {v:,}")
