"""
nanomind/eval/generation_quality.py — Heuristic generation quality metrics.

These are lightweight, reference-free metrics that give a quick sense
of text quality without needing ground-truth references.
"""

from __future__ import annotations

import math
from collections import Counter


def type_token_ratio(text: str) -> float:
    """
    Compute the Type-Token Ratio (TTR) — a simple lexical diversity metric.

    TTR = unique_words / total_words

    A higher TTR indicates greater vocabulary diversity.
    Repetitive models tend to have low TTR.

    Args:
        text: Generated text string.

    Returns:
        TTR as a float in [0, 1].
    """
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def distinct_n(text: str, n: int = 2) -> float:
    """
    Compute distinct-N: fraction of unique N-grams over all N-grams.

    A higher distinct-N means less repetition in generated text.

    Args:
        text: Generated text string.
        n:    N-gram order.

    Returns:
        Distinct-N as a float in [0, 1].
    """
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def repetition_fraction(text: str, n: int = 4) -> float:
    """
    Estimate the fraction of N-grams that are repeated (appear > once).

    Higher values = more repetition (worse generation quality).

    Args:
        text: Generated text string.
        n:    N-gram order.

    Returns:
        Repetition fraction in [0, 1].
    """
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(v for v in counts.values() if v > 1)
    return repeated / max(len(ngrams), 1)


def generation_report(text: str) -> dict:
    """
    Compute a full generation quality report for a text string.

    Returns:
        Dict with ``ttr``, ``distinct_1``, ``distinct_2``, ``repetition_4``,
        ``n_words``, ``n_chars`` fields.
    """
    return {
        "ttr":           type_token_ratio(text),
        "distinct_1":    distinct_n(text, n=1),
        "distinct_2":    distinct_n(text, n=2),
        "repetition_4":  repetition_fraction(text, n=4),
        "n_words":       len(text.split()),
        "n_chars":       len(text),
    }
