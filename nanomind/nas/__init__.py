"""NanoMind NAS sub-package — Neural Architecture Search.

Implements the full NAS pipeline from search space definition
through proxy evaluation to architecture selection:

  1. SearchSpace / ArchConfig  — discrete HP choices, sampling
  2. ProxyEvaluator            — param count, synaptic flow, loss proxy
  3. RandomSearch              — random sampling baseline
  4. EvolutionarySearch        — tournament + mutation
  5. ProgressiveScheduler      — shrink space over rounds
  6. Supernet                  — weight-sharing one-shot NAS
  7. pareto_front()            — multi-objective Pareto analysis

Primary exports:
    - :class:`ArchConfig`            — d_model, n_layers, n_heads, ffn_ratio
    - :class:`SearchSpace`           — random_sample, grid, neighbours, size
    - :class:`ProxyEvaluator`        — evaluate() → composite score
    - :class:`SearchResult`          — config + scores + rank
    - :class:`RandomSearch`          — run(), top_k(), summary()
    - :class:`EvolutionarySearch`    — run(), generation_bests(), summary()
    - :class:`ProgressiveScheduler`  — get_space(), temperature()
    - :class:`WarmRestartScheduler`  — step(), reset()
    - :class:`Supernet`              — weight-sharing forward, sample_subnet()
    - :class:`SupernetBlock`         — elastic-width transformer block
    - :func:`pareto_front`           — non-dominated architecture set
    - :func:`efficiency_score`       — accuracy-vs-params trade-off
    - :class:`ParetoPoint`           — Pareto front point
"""

from nanomind.nas.search_space import ArchConfig, SearchSpace
from nanomind.nas.evaluator import ProxyEvaluator
from nanomind.nas.random_search import RandomSearch, SearchResult
from nanomind.nas.evolutionary import EvolutionarySearch
from nanomind.nas.scheduler import ProgressiveScheduler, WarmRestartScheduler
from nanomind.nas.supernet import Supernet, SupernetBlock
from nanomind.nas.pareto import pareto_front, ParetoPoint, efficiency_score

__all__ = [
    "ArchConfig", "SearchSpace",
    "ProxyEvaluator",
    "RandomSearch", "SearchResult",
    "EvolutionarySearch",
    "ProgressiveScheduler", "WarmRestartScheduler",
    "Supernet", "SupernetBlock",
    "pareto_front", "ParetoPoint", "efficiency_score",
]
