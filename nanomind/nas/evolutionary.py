"""
nanomind/nas/evolutionary.py — Evolutionary/genetic architecture search.

Evolutionary NAS mimics natural selection:
  1. Initialise: random population of N architectures
  2. Evaluate: score all with proxy evaluator
  3. Select:   keep top-K (tournament or truncation selection)
  4. Mutate:   modify one hyperparameter per child
  5. Repeat for G generations

Used by: AmoebaNet (Real et al., 2019), EfficientNet-NAS.

Compared to random search:
  - More sample-efficient (guided by fitness signal)
  - Can exploit structure (good configs produce good children)
  - Risk: premature convergence to local optima

Reference:
  Real et al. (2019) "Regularized Evolution for Image Classifier Architecture Search"
  https://arxiv.org/abs/1802.01548
"""

from __future__ import annotations
import random
import copy
from nanomind.nas.search_space import SearchSpace, ArchConfig
from nanomind.nas.evaluator import ProxyEvaluator
from nanomind.nas.random_search import SearchResult
from nanomind.utils.logger import get_logger

log = get_logger("nas.evolutionary")


def _mutate(cfg: ArchConfig, space: SearchSpace) -> ArchConfig:
    """Mutate one randomly chosen hyperparameter."""
    neighbours = space.neighbours(cfg, n=1)
    return neighbours[0] if neighbours else space.random_sample()


class EvolutionarySearch:
    """
    Evolutionary NAS with tournament selection and mutation.

    Args:
        space:       Search space.
        evaluator:   Proxy evaluator.
        population:  Population size.
        generations: Number of evolutionary generations.
        top_k:       Number of elites kept each generation.
        tournament:  Tournament size for parent selection.
        seed:        Random seed.

    Example::

        search = EvolutionarySearch(space, evaluator, population=10, generations=5)
        best   = search.run()
    """

    def __init__(
        self,
        space:       SearchSpace,
        evaluator:   ProxyEvaluator,
        population:  int = 10,
        generations: int = 5,
        top_k:       int = 5,
        tournament:  int = 3,
        seed:        int = 0,
    ) -> None:
        self.space       = space
        self.evaluator   = evaluator
        self.population  = population
        self.generations = generations
        self.top_k       = top_k
        self.tournament  = tournament
        random.seed(seed)
        self._history:    list[list[SearchResult]] = []
        self._all_results: list[SearchResult]      = []

    def _tournament_select(
        self, pool: list[SearchResult]
    ) -> SearchResult:
        """Pick best from a random tournament subset."""
        contestants = random.sample(pool, min(self.tournament, len(pool)))
        return max(contestants, key=lambda r: r.composite)

    def _evaluate(self, cfg: ArchConfig) -> SearchResult:
        scores = self.evaluator.evaluate(cfg)
        return SearchResult(config=cfg, scores=scores)

    def run(self, verbose: bool = False) -> SearchResult:
        """
        Run evolutionary search.

        Returns:
            Best :class:`SearchResult` found across all generations.
        """
        self._history.clear()
        self._all_results.clear()

        # Initialise population
        pop = [self._evaluate(self.space.random_sample())
               for _ in range(self.population)]
        self._all_results.extend(pop)

        for gen in range(self.generations):
            pop.sort(key=lambda r: r.composite, reverse=True)
            self._history.append(list(pop))
            if verbose:
                best = pop[0]
                log.info(f"Gen {gen+1}/{self.generations}  "
                         f"best={best.composite:.4f}  "
                         f"d={best.config.d_model} L={best.config.n_layers}")

            # Select elites + generate children
            elites   = pop[:self.top_k]
            children = []
            while len(children) < self.population - self.top_k:
                parent = self._tournament_select(elites)
                child  = _mutate(parent.config, self.space)
                result = self._evaluate(child)
                children.append(result)
                self._all_results.append(result)

            pop = elites + children

        # Final sort
        self._all_results.sort(key=lambda r: r.composite, reverse=True)
        for rank, r in enumerate(self._all_results):
            r.rank = rank + 1
        return self._all_results[0]

    @property
    def best(self) -> SearchResult | None:
        return self._all_results[0] if self._all_results else None

    def generation_bests(self) -> list[float]:
        """Best composite score per generation."""
        return [max(r.composite for r in gen) for gen in self._history]

    def summary(self) -> dict:
        if not self._all_results:
            return {}
        scores = [r.composite for r in self._all_results]
        return {
            "n_evaluated":   len(self._all_results),
            "generations":   self.generations,
            "best_score":    max(scores),
            "best_config":   self._all_results[0].config.to_dict(),
            "gen_bests":     self.generation_bests(),
        }
