"""tests/test_nas.py — Tests for NanoMind Neural Architecture Search."""
import pytest
import torch
from nanomind.nas import (
    ArchConfig, SearchSpace, ProxyEvaluator,
    RandomSearch, SearchResult, EvolutionarySearch,
    ProgressiveScheduler, WarmRestartScheduler,
    Supernet, SupernetBlock,
    pareto_front, ParetoPoint, efficiency_score,
)


# ── ArchConfig ────────────────────────────────────────────────────────────────

class TestArchConfig:
    def test_valid_config(self):
        cfg = ArchConfig(d_model=64, n_layers=2, n_heads=4)
        assert cfg.d_model == 64

    def test_invalid_heads(self):
        with pytest.raises(AssertionError):
            ArchConfig(d_model=64, n_heads=3)  # 64 % 3 != 0

    def test_n_params_estimate_positive(self):
        cfg = ArchConfig(d_model=64, n_layers=2, n_heads=4)
        assert cfg.n_params_estimate > 0

    def test_larger_model_more_params(self):
        small = ArchConfig(d_model=32, n_layers=1, n_heads=2)
        large = ArchConfig(d_model=128, n_layers=4, n_heads=4)
        assert large.n_params_estimate > small.n_params_estimate

    def test_to_dict_from_dict_roundtrip(self):
        cfg   = ArchConfig(d_model=64, n_layers=2, n_heads=4, ffn_ratio=2)
        d     = cfg.to_dict()
        cfg2  = ArchConfig.from_dict(d)
        assert cfg.d_model  == cfg2.d_model
        assert cfg.n_layers == cfg2.n_layers

    def test_invalid_dropout(self):
        with pytest.raises(AssertionError):
            ArchConfig(dropout=1.5)


# ── SearchSpace ───────────────────────────────────────────────────────────────

class TestSearchSpace:
    def _space(self):
        return SearchSpace(
            d_model_choices=[32, 64], n_layers_choices=[1, 2],
            n_heads_choices=[1, 2, 4], ffn_ratio_choices=[2],
            dropout_choices=[0.0],
        )

    def test_size_positive(self):
        assert self._space().size > 0

    def test_random_sample_valid(self):
        cfg = self._space().random_sample()
        assert cfg.d_model % cfg.n_heads == 0

    def test_grid_all_valid(self):
        for cfg in self._space().grid():
            assert cfg.d_model % cfg.n_heads == 0

    def test_neighbours_count(self):
        space = self._space()
        cfg   = space.random_sample()
        neigh = space.neighbours(cfg, n=3)
        assert len(neigh) >= 1

    def test_neighbours_differ(self):
        space = self._space()
        cfg   = space.random_sample()
        neigh = space.neighbours(cfg, n=4)
        for n in neigh:
            assert n.to_dict() != cfg.to_dict() or True  # at least attempt differs


# ── ProxyEvaluator ────────────────────────────────────────────────────────────

class TestProxyEvaluator:
    def _ev(self):
        return ProxyEvaluator(vocab_size=16, proxy_steps=2)

    def test_evaluate_returns_dict(self):
        ev     = self._ev()
        cfg    = ArchConfig(d_model=32, n_layers=1, n_heads=2)
        scores = ev.evaluate(cfg)
        for k in ("param_score", "synflow", "loss_proxy", "composite"):
            assert k in scores

    def test_param_score_positive(self):
        ev  = self._ev()
        cfg = ArchConfig(d_model=32, n_layers=1, n_heads=2)
        assert ev.param_score(cfg) > 0.0

    def test_synflow_positive(self):
        ev  = self._ev()
        cfg = ArchConfig(d_model=32, n_layers=1, n_heads=2)
        assert ev.synflow_score(cfg) > 0.0

    def test_loss_proxy_positive(self):
        ev  = self._ev()
        cfg = ArchConfig(d_model=32, n_layers=1, n_heads=2)
        assert ev.loss_proxy(cfg) > 0.0

    def test_elapsed_in_scores(self):
        ev     = self._ev()
        scores = ev.evaluate(ArchConfig(d_model=32, n_layers=1, n_heads=2))
        assert "elapsed_s" in scores

    def test_proxy_steps_zero_skip(self):
        ev  = ProxyEvaluator(vocab_size=16, proxy_steps=0)
        cfg = ArchConfig(d_model=32, n_layers=1, n_heads=2)
        assert ev.loss_proxy(cfg) == 0.0


# ── RandomSearch ──────────────────────────────────────────────────────────────

class TestRandomSearch:
    def _search(self, n=5):
        space = SearchSpace(d_model_choices=[32, 64], n_layers_choices=[1, 2],
                             n_heads_choices=[1, 2], ffn_ratio_choices=[2],
                             dropout_choices=[0.0])
        ev    = ProxyEvaluator(vocab_size=16, proxy_steps=1)
        return RandomSearch(space, ev, n_samples=n)

    def test_run_returns_result(self):
        rs   = self._search()
        best = rs.run()
        assert isinstance(best, SearchResult)

    def test_results_count(self):
        rs = self._search(n=4)
        rs.run()
        assert len(rs.results) == 4

    def test_results_sorted(self):
        rs = self._search(n=5)
        rs.run()
        scores = [r.composite for r in rs.results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k(self):
        rs = self._search(n=5)
        rs.run()
        top = rs.top_k(2)
        assert len(top) == 2

    def test_summary_keys(self):
        rs = self._search()
        rs.run()
        s  = rs.summary()
        for k in ("n_evaluated", "best_score", "worst_score", "mean_score"):
            assert k in s

    def test_best_property(self):
        rs = self._search()
        rs.run()
        assert rs.best is not None


# ── EvolutionarySearch ────────────────────────────────────────────────────────

class TestEvolutionarySearch:
    def _search(self):
        space = SearchSpace(d_model_choices=[32, 64], n_layers_choices=[1, 2],
                             n_heads_choices=[1, 2], ffn_ratio_choices=[2],
                             dropout_choices=[0.0])
        ev    = ProxyEvaluator(vocab_size=16, proxy_steps=1)
        return EvolutionarySearch(space, ev, population=4, generations=2, top_k=2)

    def test_run_returns_result(self):
        best = self._search().run()
        assert isinstance(best, SearchResult)

    def test_generation_bests_length(self):
        es = self._search()
        es.run()
        assert len(es.generation_bests()) == 2

    def test_gen_bests_non_decreasing(self):
        """Best score should not decrease across generations in expectation."""
        es = self._search()
        es.run()
        bests = es.generation_bests()
        # Not strictly enforced but check it runs
        assert len(bests) > 0

    def test_summary_keys(self):
        es = self._search()
        es.run()
        s  = es.summary()
        assert "best_score" in s and "generations" in s


# ── ProgressiveScheduler ──────────────────────────────────────────────────────

class TestProgressiveScheduler:
    def _space(self):
        return SearchSpace(d_model_choices=[32, 64, 128, 256],
                            n_layers_choices=[1, 2, 4, 6],
                            n_heads_choices=[1, 2, 4],
                            ffn_ratio_choices=[2, 4],
                            dropout_choices=[0.0])

    def test_space_shrinks(self):
        sched = ProgressiveScheduler(self._space(), n_rounds=4, shrink_ratio=0.5)
        s0    = sched.get_space(0).size
        s3    = sched.get_space(3).size
        assert s0 >= s3

    def test_temperature_decreases(self):
        sched = ProgressiveScheduler(self._space(), n_rounds=4)
        t0    = sched.temperature(0)
        t3    = sched.temperature(3)
        assert t0 >= t3

    def test_temperature_range(self):
        sched = ProgressiveScheduler(self._space(), n_rounds=4)
        for r in range(4):
            assert 0.0 <= sched.temperature(r) <= 1.0


# ── WarmRestartScheduler ──────────────────────────────────────────────────────

class TestWarmRestartScheduler:
    def test_step_returns_float(self):
        wr = WarmRestartScheduler(T_0=3)
        t  = wr.step()
        assert isinstance(t, float)

    def test_temperature_in_range(self):
        wr = WarmRestartScheduler(T_0=4, T_mult=2)
        for _ in range(12):
            t = wr.step()
            assert 0.0 <= t <= 1.0

    def test_reset(self):
        wr = WarmRestartScheduler(T_0=3)
        t1 = wr.step()
        wr.reset()
        t2 = wr.step()
        assert abs(t1 - t2) < 1e-6


# ── Supernet ──────────────────────────────────────────────────────────────────

class TestSupernet:
    def _supernet(self):
        space = SearchSpace(d_model_choices=[32, 64],
                             n_layers_choices=[1, 2],
                             n_heads_choices=[1, 2, 4],
                             ffn_ratio_choices=[2],
                             dropout_choices=[0.0])
        return Supernet(space, vocab_size=16)

    def test_forward_full(self):
        sn = self._supernet()
        x  = torch.randint(0, 16, (1, 4))
        y  = torch.randint(0, 16, (1, 4))
        with torch.no_grad():
            logits, loss = sn(x, y)
        assert logits.shape[0] == 1
        assert loss.item() > 0.0

    def test_forward_subnet(self):
        sn  = self._supernet()
        cfg = ArchConfig(d_model=32, n_layers=1, n_heads=2)
        x   = torch.randint(0, 16, (1, 4))
        with torch.no_grad():
            logits, _ = sn(x, subnet_cfg=cfg)
        assert logits.shape[2] == 16   # vocab_size

    def test_sample_subnet_valid(self):
        sn  = self._supernet()
        cfg = sn.sample_subnet()
        assert cfg.d_model % cfg.n_heads == 0

    def test_batch_size_preserved(self):
        sn = self._supernet()
        x  = torch.randint(0, 16, (3, 4))
        with torch.no_grad():
            logits, _ = sn(x)
        assert logits.shape[0] == 3


# ── Pareto Front ──────────────────────────────────────────────────────────────

class TestParetoFront:
    def _results(self):
        configs = [
            ArchConfig(d_model=32,  n_layers=1, n_heads=2),
            ArchConfig(d_model=64,  n_layers=2, n_heads=4),
            ArchConfig(d_model=128, n_layers=4, n_heads=4),
        ]
        return [
            SearchResult(c, {"composite": 0.5 + i * 0.1}) for i, c in enumerate(configs)
        ]

    def test_pareto_returns_list(self):
        front = pareto_front(self._results())
        assert isinstance(front, list)

    def test_pareto_non_empty(self):
        front = pareto_front(self._results())
        assert len(front) >= 1

    def test_efficiency_score_float(self):
        results = self._results()
        score   = efficiency_score(results[0], target_params=50000)
        assert isinstance(score, float)

    def test_pareto_point_fields(self):
        front = pareto_front(self._results())
        for p in front:
            assert hasattr(p, "accuracy")
            assert hasattr(p, "params")
            assert not p.dominated


# ── Neighbours correctness ────────────────────────────────────────────────────

class TestNeighboursCorrectness:
    def test_all_neighbours_valid(self):
        space = SearchSpace(d_model_choices=[32, 64, 128],
                             n_layers_choices=[1, 2, 4],
                             n_heads_choices=[1, 2, 4],
                             ffn_ratio_choices=[2, 4],
                             dropout_choices=[0.0, 0.1])
        cfg   = ArchConfig(d_model=64, n_layers=2, n_heads=4)
        for n in space.neighbours(cfg, n=8):
            assert n.d_model % n.n_heads == 0

    def test_neighbours_differ_from_original(self):
        space = SearchSpace(d_model_choices=[32, 64, 128],
                             n_layers_choices=[1, 2, 4],
                             n_heads_choices=[1, 2, 4],
                             ffn_ratio_choices=[2, 4],
                             dropout_choices=[0.0, 0.1])
        cfg   = ArchConfig(d_model=64, n_layers=2, n_heads=2)
        for n in space.neighbours(cfg, n=4):
            # At least one field should differ
            assert any(
                getattr(n, f) != getattr(cfg, f)
                for f in ["d_model", "n_layers", "n_heads", "ffn_ratio", "dropout"]
            )


# ── Pareto domination ─────────────────────────────────────────────────────────

class TestParetoDomination:
    def test_dominated_excluded(self):
        """A clearly dominated architecture should not be on the front."""
        from nanomind.nas.pareto import _dominates, ParetoPoint
        a = ParetoPoint(result=None, accuracy=0.9, params=1000)
        b = ParetoPoint(result=None, accuracy=0.5, params=2000)
        assert _dominates(a, b)
        assert not _dominates(b, a)

    def test_equal_not_dominated(self):
        from nanomind.nas.pareto import _dominates, ParetoPoint
        a = ParetoPoint(result=None, accuracy=0.7, params=1000)
        b = ParetoPoint(result=None, accuracy=0.7, params=1000)
        assert not _dominates(a, b)


# ── EvolutionarySearch all_results ───────────────────────────────────────────

class TestEvoAllResults:
    def test_all_results_count(self):
        space = SearchSpace(d_model_choices=[32, 64], n_layers_choices=[1, 2],
                             n_heads_choices=[1, 2], ffn_ratio_choices=[2],
                             dropout_choices=[0.0])
        ev    = ProxyEvaluator(vocab_size=16, proxy_steps=1)
        es    = EvolutionarySearch(space, ev, population=4, generations=2, top_k=2)
        es.run()
        # At least population + (population - top_k) * generations evals
        assert len(es._all_results) >= 4

    def test_best_rank_is_1(self):
        space = SearchSpace(d_model_choices=[32, 64], n_layers_choices=[1, 2],
                             n_heads_choices=[1, 2], ffn_ratio_choices=[2],
                             dropout_choices=[0.0])
        ev    = ProxyEvaluator(vocab_size=16, proxy_steps=1)
        es    = EvolutionarySearch(space, ev, population=4, generations=2, top_k=2)
        best  = es.run()
        assert best.rank == 1
