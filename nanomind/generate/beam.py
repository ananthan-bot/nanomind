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
