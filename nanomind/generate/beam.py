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
