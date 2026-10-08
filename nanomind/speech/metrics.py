"""
nanomind/speech/metrics.py — Word Error Rate (WER), Character Error Rate (CER), and Codebook Perplexity.
"""
import math
from typing import List, Dict, Any
import torch


def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """Compute Levenshtein edit distance between two token sequences."""
    n1, n2 = len(seq1), len(seq2)
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]

    for i in range(n1 + 1):
        dp[i][0] = i
    for j in range(n2 + 1):
        dp[0][j] = j

    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n1][n2]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) = (Substitutions + Deletions + Insertions) / Total Reference Words.
    """
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return float(dist / len(ref_words))


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) at character level."""
    ref_chars = list(reference.strip().lower())
    hyp_chars = list(hypothesis.strip().lower())

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    dist = levenshtein_distance(ref_chars, hyp_chars)
    return float(dist / len(ref_chars))


def compute_codebook_perplexity(codes: torch.Tensor, codebook_size: int = 1024) -> float:
    """
    Measures the perplexity / utilization of codebook entries in RVQ:
    Perplexity = exp(- sum p_k * log(p_k))
    A higher perplexity (closer to codebook_size) indicates uniform utilization (no codebook collapse).
    """
    flat = codes.flatten()
    if len(flat) == 0:
        return 0.0

    counts = torch.bincount(flat, minlength=codebook_size).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log(probs))
    return float(torch.exp(entropy).item())
