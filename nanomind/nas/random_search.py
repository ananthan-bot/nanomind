"""
nanomind/nas/random_search.py — Random architecture search.

Random search is a surprisingly strong NAS baseline.
Bergstra & Bengio (2012) showed random search beats grid search
because it explores the space more efficiently:
  - Grid search repeats similar configs
  - Random search always tries new combinations

Runtime: O(N) evaluations, fully parallelisable.

Reference:
  Bergstra & Bengio (2012) "Random Search for Hyper-Parameter Optimization"
  https://jmlr.csail.mit.edu/papers/v13/bergstra12a.html
"""

from __future__ import annotations
import random
from dataclasses import dataclass

from nanomind.nas.search_space import SearchSpace, ArchConfig
from nanomind.nas.evaluator import ProxyEvaluator
from nanomind.utils.logger import get_logger

log = get_logger("nas.random_search")


@dataclass
class SearchResult:
    """Result of a single architecture evaluation."""
    config:  ArchConfig
    scores:  dict
    rank:    int = 0

    @property
    def composite(self) -> float:
        return self.scores.get("composite", 0.0)


class RandomSearch:
    """
    Random architecture search over a :class:`SearchSpace`.

    Args:
        space:     Architecture search space.
        evaluator: Proxy evaluator.
        n_samples: Number of random architectures to evaluate.
        seed:      Random seed for reproducibility.

    Example::

        search = RandomSearch(SearchSpace(), ProxyEvaluator(), n_samples=20)
        best   = search.run()
        print(best.config)
    """

    def __init__(
        self,
        space:     SearchSpace,
        evaluator: ProxyEvaluator,
        n_samples: int = 20,
        seed:      int = 0,
    ) -> None:
        self.space     = space
        self.evaluator = evaluator
        self.n_samples = n_samples
        random.seed(seed)
        self._results: list[SearchResult] = []

    def run(self, verbose: bool = False) -> SearchResult:
        """
        Run random search and return the best architecture.

        Args:
            verbose: Print progress.

        Returns:
            Best :class:`SearchResult`.
        """
        self._results.clear()
        for i in range(self.n_samples):
            cfg    = self.space.random_sample()
            scores = self.evaluator.evaluate(cfg)
            result = SearchResult(config=cfg, scores=scores)
            self._results.append(result)
            if verbose:
                log.info(f"[{i+1}/{self.n_samples}] "
                         f"composite={scores['composite']:.4f} "
                         f"d={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads}")

        # Rank by composite score (descending)
        self._results.sort(key=lambda r: r.composite, reverse=True)
        for rank, r in enumerate(self._results):
            r.rank = rank + 1
        return self._results[0]

    @property
    def results(self) -> list[SearchResult]:
        return list(self._results)

    @property
    def best(self) -> SearchResult | None:
        return self._results[0] if self._results else None

    def top_k(self, k: int = 3) -> list[SearchResult]:
        """Return top-K results by composite score."""
        return self._results[:k]

    def summary(self) -> dict:
        """Summary statistics of the search."""
        if not self._results:
            return {}
        scores = [r.composite for r in self._results]
        return {
            "n_evaluated":  len(self._results),
            "best_score":   max(scores),
            "worst_score":  min(scores),
            "mean_score":   sum(scores) / len(scores),
            "best_config":  self._results[0].config.to_dict(),
        }
