"""
nanomind/nas/pareto.py — Multi-objective Pareto front for NAS.

Real-world NAS has multiple objectives:
  - Maximise accuracy (composite proxy score)
  - Minimise latency / parameter count / FLOPs

Pareto front: the set of architectures where you cannot improve
one objective without worsening another.

Used by: MnasNet (Tan et al., 2018), EfficientNet, ProxylessNAS.

Reference:
  MnasNet: Tan et al. (2018) https://arxiv.org/abs/1807.11626
  NSGA-II: Deb et al. (2002) — standard multi-objective EA
"""

from __future__ import annotations
from dataclasses import dataclass
from nanomind.nas.random_search import SearchResult


@dataclass
class ParetoPoint:
    """A point on the Pareto front."""
    result:    SearchResult
    accuracy:  float    # composite proxy score
    params:    int      # parameter estimate
    dominated: bool = False


def _dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """Return True if a dominates b (better or equal on all objectives)."""
    return (a.accuracy >= b.accuracy and a.params <= b.params
            and (a.accuracy > b.accuracy or a.params < b.params))


def pareto_front(results: list[SearchResult]) -> list[ParetoPoint]:
    """
    Compute the Pareto front from a list of search results.

    Objectives:
      - Maximise composite proxy score (accuracy proxy)
      - Minimise parameter count (efficiency)

    Args:
        results: List of :class:`SearchResult`.

    Returns:
        List of non-dominated :class:`ParetoPoint` objects.
    """
    points = [
        ParetoPoint(
            result   = r,
            accuracy = r.composite,
            params   = r.config.n_params_estimate,
        )
        for r in results
    ]

    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i != j and _dominates(q, p):
                p.dominated = True
                break

    front = [p for p in points if not p.dominated]
    front.sort(key=lambda p: p.params)
    return front


def efficiency_score(result: SearchResult, target_params: int) -> float:
    """
    Score architecture by accuracy-vs-params trade-off.

    Args:
        result:        Search result.
        target_params: Target parameter budget.

    Returns:
        Score that rewards being close to target while being accurate.
    """
    param_penalty = abs(result.config.n_params_estimate - target_params) / max(target_params, 1)
    return result.composite - 0.1 * param_penalty
