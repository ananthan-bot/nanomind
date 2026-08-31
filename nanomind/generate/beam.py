"""
nanomind/generate/beam.py — Beam Search and Diverse Beam Search.

Greedy decoding always takes the single most likely token at each step.
Beam search keeps the top-B most likely *sequences*, expanding each one
at every step and pruning back to B beams:

  Greedy:   O(T)     sequences explored — fast but suboptimal
  Beam (B): O(T × B) sequences explored — better quality, B times slower

Diverse Beam Search (Vijayakumar et al. 2016) splits the B beams into G groups
and penalises each group for choosing tokens already chosen by earlier groups,
encouraging diversity across the returned hypotheses.

References:
  Beam search:          Sutskever et al. (2014)
  Diverse beam search:  Vijayakumar et al. (2016) — https://arxiv.org/abs/1610.02424
  Length penalty:       Wu et al. (2016) GNMT — https://arxiv.org/abs/1609.08144
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BeamConfig:
    """
    Configuration for beam search generation.

    Attributes:
        num_beams:          Number of beams (width of the search).
                            1 = greedy decoding, >1 = beam search.
        max_new_tokens:     Maximum tokens to generate per beam.
        length_penalty:     Exponent for length normalisation.
                            α > 1 favours longer sequences; α < 1 shorter.
                            α = 1.0 means no length penalty.
        early_stopping:     Stop when all beams have produced EOS.
        no_repeat_ngram:    Block repeated n-grams (0 = off).
        num_beam_groups:    Number of diverse beam groups (Diverse BS).
                            Must divide ``num_beams`` evenly.
        diversity_penalty:  Penalty for tokens chosen by earlier groups (DBS).
        temperature:        Logit temperature before beam scoring.
        top_k:              Top-K filter on logits (0 = off).
        return_n_best:      How many final hypotheses to return (≤ num_beams).
    """

    num_beams:         int   = 4
    max_new_tokens:    int   = 50
    length_penalty:    float = 1.0
    early_stopping:    bool  = True
    no_repeat_ngram:   int   = 0
    num_beam_groups:   int   = 1
    diversity_penalty: float = 0.5
    temperature:       float = 1.0
    top_k:             int   = 0
    return_n_best:     int   = 1

    def __post_init__(self) -> None:
        assert self.num_beams >= 1
        assert self.num_beams % self.num_beam_groups == 0,             "num_beams must be divisible by num_beam_groups"
        assert self.return_n_best <= self.num_beams
        assert self.temperature > 0.0
        assert self.max_new_tokens > 0


import torch
import torch.nn.functional as F
import math


class BeamHypothesis:
    """
    A single beam hypothesis: a sequence of tokens with an accumulated log-prob.

    Args:
        tokens:   Token IDs generated so far (including prompt).
        log_prob: Cumulative log-probability (unnormalized).
    """

    __slots__ = ("tokens", "log_prob")

    def __init__(self, tokens: list[int], log_prob: float = 0.0) -> None:
        self.tokens   = tokens
        self.log_prob = log_prob

    def score(self, length_penalty: float = 1.0) -> float:
        """Length-normalised score (GNMT formula)."""
        lp = ((5 + len(self.tokens)) / 6) ** length_penalty
        return self.log_prob / lp

    def extend(self, token_id: int, log_p: float) -> "BeamHypothesis":
        return BeamHypothesis(self.tokens + [token_id], self.log_prob + log_p)

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        return f"BeamHypothesis(len={len(self.tokens)}, score={self.score():.3f})"


class BeamHypotheses:
    """
    Container for completed beam hypotheses.

    Keeps only the top ``num_beams`` completed hypotheses by length-penalised score.

    Args:
        num_beams:      Maximum number of hypotheses to store.
        length_penalty: GNMT length penalty exponent.
        early_stopping: Whether to declare done when filled.
    """

    def __init__(
        self,
        num_beams:      int,
        length_penalty: float = 1.0,
        early_stopping: bool  = True,
    ) -> None:
        self.num_beams      = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.hyps: list[BeamHypothesis] = []

    def add(self, hyp: BeamHypothesis) -> None:
        """Add a completed hypothesis, keeping only the top num_beams."""
        self.hyps.append(hyp)
        self.hyps.sort(key=lambda h: h.score(self.length_penalty), reverse=True)
        if len(self.hyps) > self.num_beams:
            self.hyps.pop()

    @property
    def is_done(self) -> bool:
        """True if we have enough hypotheses and early stopping is enabled."""
        return self.early_stopping and len(self.hyps) >= self.num_beams

    def best(self, n: int = 1) -> list[BeamHypothesis]:
        """Return the top-n hypotheses by score."""
        return self.hyps[:n]


import torch.nn as nn


def _get_next_logprobs(
    model:       nn.Module,
    input_ids:   torch.Tensor,
    temperature: float = 1.0,
    top_k:       int   = 0,
    block_size:  int   = 512,
) -> torch.Tensor:
    """
    Run model forward and return log-probabilities for the next token.

    Args:
        model:      Language model.
        input_ids:  ``(1, T)`` token IDs.
        temperature: Temperature for logit scaling.
        top_k:      Top-K filtering (0 = off).
        block_size: Model's context window.

    Returns:
        Log-prob tensor ``(vocab_size,)``.
    """
    ctx    = input_ids[:, -block_size:]
    logits, _ = model(ctx)
    logits = logits[0, -1, :].float() / max(temperature, 1e-8)

    if top_k > 0:
        topk_vals = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < topk_vals, float("-inf"))

    return F.log_softmax(logits, dim=-1)


def _block_repeat_ngrams(
    tokens:       list[int],
    log_probs:    torch.Tensor,
    no_repeat_ngram: int,
) -> torch.Tensor:
    """
    Block tokens that would create a repeated n-gram.

    Scans the current token sequence for any n-gram of length ``no_repeat_ngram``
    and sets the log-prob of any token that would continue a seen n-gram to -inf.

    Args:
        tokens:          Current token sequence.
        log_probs:       Next-token log-probs ``(vocab_size,)``.
        no_repeat_ngram: Block n-grams of this length (0 = off).

    Returns:
        Modified log-probs with blocked tokens set to -inf.
    """
    if no_repeat_ngram <= 0 or len(tokens) < no_repeat_ngram:
        return log_probs

    n   = no_repeat_ngram
    lp  = log_probs.clone()
    # Build set of banned next tokens
    banned: set[int] = set()
    tail = tokens[-(n - 1):]   # last (n-1) tokens
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n - 1] == tail:
            banned.add(tokens[i + n - 1])
    for tok in banned:
        lp[tok] = float("-inf")
    return lp


@torch.no_grad()
def beam_search(
    model:        nn.Module,
    input_ids:    torch.Tensor,
    cfg:          BeamConfig | None = None,
    eos_token_id: int | None = None,
) -> list[BeamHypothesis]:
    """
    Standard beam search generation.

    Maintains ``num_beams`` candidate sequences in parallel, expanding each
    by the top-``num_beams`` next tokens and keeping the globally best B sequences.

    Args:
        model:        Language model with forward(input_ids) → (logits, _).
        input_ids:    Prompt token IDs ``(1, T)``.
        cfg:          Beam search configuration.
        eos_token_id: End-of-sequence token ID.

    Returns:
        List of completed :class:`BeamHypothesis` objects sorted by score.
        Length is ``cfg.return_n_best``.
    """
    cfg   = cfg or BeamConfig()
    model.eval()
    device      = input_ids.device
    block_size  = getattr(model, "cfg", None) and model.cfg.block_size or 512
    vocab_size  = getattr(model, "cfg", None) and model.cfg.vocab_size or model.lm_head.out_features

    # Initialise beams: one starting hypothesis
    prompt_tokens = input_ids[0].tolist()
    active_beams  = [BeamHypothesis(prompt_tokens, 0.0)]
    completed     = BeamHypotheses(cfg.num_beams, cfg.length_penalty, cfg.early_stopping)

    for _ in range(cfg.max_new_tokens):
        if not active_beams:
            break

        candidates: list[tuple[float, BeamHypothesis]] = []

        for hyp in active_beams:
            ids    = torch.tensor([hyp.tokens], dtype=torch.long, device=device)
            lp     = _get_next_logprobs(model, ids, cfg.temperature, cfg.top_k, block_size)

            if cfg.no_repeat_ngram > 0:
                lp = _block_repeat_ngrams(hyp.tokens, lp, cfg.no_repeat_ngram)

            # Expand: take top num_beams tokens
            top_lp, top_ids = torch.topk(lp, min(cfg.num_beams, vocab_size))
            for tok, tok_lp in zip(top_ids.tolist(), top_lp.tolist()):
                new_hyp = hyp.extend(tok, tok_lp)
                candidates.append((new_hyp.log_prob, new_hyp))

        # Sort candidates globally by cumulative log-prob
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Partition into complete (EOS) and active beams
        active_beams = []
        for _, hyp in candidates:
            if eos_token_id is not None and hyp.tokens[-1] == eos_token_id:
                completed.add(hyp)
                if completed.is_done:
                    break
            else:
                active_beams.append(hyp)
                if len(active_beams) >= cfg.num_beams:
                    break

        if completed.is_done:
            break

    # Add any remaining active beams as completed
    for hyp in active_beams:
        completed.add(hyp)

    results = completed.best(cfg.return_n_best)
    if not results:
        results = active_beams[:cfg.return_n_best]
    return results
