"""
nanomind/eval/metrics.py — Evaluation metrics for LLM benchmarks.

## Metrics for Different Task Types

Multiple Choice:
  - Accuracy: fraction of correct answers
  - Normalized accuracy: corrected for chance (macro-averaged across subjects)

Open Generation:
  - Exact Match (EM): does prediction == ground truth (after normalization)?
  - F1: token-level overlap between prediction and reference
  - ROUGE-L: longest common subsequence F1
  - BERTScore: semantic similarity via embeddings (requires BERT)

Math Reasoning:
  - Final Answer Exact Match: extract final number, check equality

Code Generation:
  - pass@k (Day 52): probability at least 1 of k samples passes tests

Instruction Following:
  - LLM-as-a-Judge: GPT-4 / Claude scores responses 1-10

## Perplexity

Measure how well a language model predicts a text corpus.
Lower perplexity = better model.

  PPL = exp(-1/N × Σ log p(xᵢ | x₁...xᵢ₋₁))

For context:
  GPT-2:    perplexity ~29 on WikiText-103
  GPT-3:    perplexity ~20
  LLaMA-70B: perplexity ~3.0 on some benchmarks
"""

from __future__ import annotations
import re
import math
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Result of a single metric computation."""
    metric_name: str
    value:       float
    n_samples:   int
    details:     dict = None

    def to_dict(self) -> dict:
        return {
            "metric":    self.metric_name,
            "value":     round(self.value, 4),
            "n_samples": self.n_samples,
        }


def exact_match(prediction: str, reference: str, normalize: bool = True) -> bool:
    """Check if prediction exactly matches reference (after normalization)."""
    if normalize:
        prediction = _normalize_text(prediction)
        reference  = _normalize_text(reference)
    return prediction == reference


def token_f1(prediction: str, reference: str) -> float:
    """
    Token-level F1 between prediction and reference.

    Used for QA tasks where exact match is too strict.
    """
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens  = _normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    pred_set = {}
    for t in pred_tokens:
        pred_set[t] = pred_set.get(t, 0) + 1
    ref_set = {}
    for t in ref_tokens:
        ref_set[t] = ref_set.get(t, 0) + 1

    common = sum(min(pred_set.get(t, 0), ref_set.get(t, 0)) for t in ref_set)
    precision = common / max(len(pred_tokens), 1)
    recall    = common / max(len(ref_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L: F1 based on Longest Common Subsequence."""
    pred = _normalize_text(prediction).split()
    ref  = _normalize_text(reference).split()
    lcs  = _lcs_length(pred, ref)
    if not pred or not ref:
        return 0.0
    p = lcs / len(pred)
    r = lcs / len(ref)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _lcs_length(a: list, b: list) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(a), len(b)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_number(text: str) -> str | None:
    """Extract the last number from a math reasoning response."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    return numbers[-1] if numbers else None


def math_exact_match(prediction: str, reference: str) -> bool:
    """Check if math answer matches (extracts final number)."""
    pred_num = extract_number(prediction)
    ref_num  = extract_number(reference)
    if pred_num is None or ref_num is None:
        return exact_match(prediction, reference)
    try:
        return abs(float(pred_num) - float(ref_num)) < 1e-6
    except ValueError:
        return pred_num == ref_num


def multiple_choice_accuracy(
    predictions: list[str],
    references:  list[str],
) -> MetricResult:
    """Compute accuracy for multiple choice predictions."""
    n       = len(predictions)
    correct = sum(1 for p, r in zip(predictions, references)
                  if p.strip().upper()[:1] == r.strip().upper()[:1])
    return MetricResult(
        metric_name = "accuracy",
        value       = correct / max(n, 1),
        n_samples   = n,
        details     = {"n_correct": correct, "n_total": n},
    )


def generation_metrics(
    predictions: list[str],
    references:  list[str],
    task_type:   str = "open",
) -> dict[str, MetricResult]:
    """Compute all generation metrics for a list of predictions."""
    n = len(predictions)
    em_scores     = [exact_match(p, r) for p, r in zip(predictions, references)]
    f1_scores     = [token_f1(p, r) for p, r in zip(predictions, references)]
    rouge_scores  = [rouge_l(p, r)   for p, r in zip(predictions, references)]

    return {
        "exact_match": MetricResult("exact_match", sum(em_scores)/n, n),
        "token_f1":    MetricResult("token_f1",    sum(f1_scores)/n, n),
        "rouge_l":     MetricResult("rouge_l",     sum(rouge_scores)/n, n),
    }


@torch.no_grad()
def perplexity(
    model:     nn.Module,
    input_ids: torch.Tensor,
    stride:    int = 512,
) -> float:
    """
    Compute perplexity of a language model on a token sequence.

    Uses strided evaluation to handle sequences longer than context window.

    Args:
        model:     LM model returning ``(B, T, V)`` logits.
        input_ids: ``(1, T)`` token IDs.
        stride:    Evaluation stride (overlap between windows).

    Returns:
        Perplexity (scalar).
    """
    import torch.nn.functional as F
    T        = input_ids.shape[1]
    nll_sum  = 0.0
    n_tokens = 0

    for begin in range(0, T - 1, stride):
        end       = min(begin + stride * 2, T)
        ids       = input_ids[:, begin:end]
        target    = ids[:, 1:]
        inp       = ids[:, :-1]

        out    = model(inp)
        logits = out[0] if isinstance(out, tuple) else out   # (1, t-1, V)
        log_p  = F.log_softmax(logits, dim=-1)

        # Gather target log probs
        tgt_lp = log_p.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (1, t-1)
        nll_sum  += -tgt_lp.sum().item()
        n_tokens += tgt_lp.numel()

    return math.exp(nll_sum / max(n_tokens, 1))
