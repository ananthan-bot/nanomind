"""
tests/test_merge.py — Comprehensive unit tests for Model Merging algorithms.
"""
import copy
import math
import torch
import torch.nn as nn
from nanomind.merge import (
    MergeConfig,
    MergeMethod,
    ModelMerger,
    TaskVector,
    TIESMerger,
    DAREPruner,
    slerp,
    ModelSoup,
    compute_parameter_drift,
    compute_task_vector_cosine_similarity,
)


class TestTaskVectors:
    def test_task_vector_arithmetic(self):
        m1 = nn.Linear(4, 2, bias=False)
        m2 = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            m1.weight.fill_(1.0)
            m2.weight.fill_(3.0)

        tv = TaskVector.from_models(m1, m2)
        assert (tv.vector_dict["weight"] == 2.0).all()

        # Addition: tv + tv -> 4.0
        tv_sum = tv + tv
        assert (tv_sum.vector_dict["weight"] == 4.0).all()

        # Scaling: tv * 0.5 -> 1.0
        tv_scaled = tv * 0.5
        assert (tv_scaled.vector_dict["weight"] == 1.0).all()

    def test_task_vector_cosine(self):
        v1 = {"w": torch.tensor([1.0, 0.0])}
        v2 = {"w": torch.tensor([0.0, 1.0])}
        sim = compute_task_vector_cosine_similarity(v1, v2)
        assert math.isclose(sim, 0.0, abs_tol=1e-5)


class TestTIES:
    def test_trimming(self):
        merger = TIESMerger(density=0.5)
        t = torch.tensor([1.0, 5.0, 2.0, 8.0])
        trimmed = merger.trim_tensor(t)
        # Top 2 values (8.0, 5.0) kept, rest zeroed
        assert trimmed[0] == 0.0
        assert trimmed[2] == 0.0
        assert trimmed[1] == 5.0
        assert trimmed[3] == 8.0

    def test_ties_merge_sign_election(self):
        merger = TIESMerger(density=1.0)
        t1 = torch.tensor([2.0, -1.0])
        t2 = torch.tensor([3.0, 1.0])
        t3 = torch.tensor([1.0, -2.0])
        # Position 0: all positive -> positive average
        # Position 1: t1 and t3 negative (-1, -2), t2 positive (+1) -> majority negative
        merged = merger.merge_tensors([t1, t2, t3])
        assert merged[0] > 0
        assert merged[1] < 0


class TestDARE:
    def test_dare_pruning_and_expectation(self):
        torch.manual_seed(0)
        pruner = DAREPruner(drop_rate=0.5, rescale=True)
        t = torch.ones(1000)
        pruned = pruner.prune_tensor(t)
        # Approximately 50% dropped
        sparsity = (pruned == 0.0).float().mean().item()
        assert 0.4 < sparsity < 0.6
        # Rescaled non-zero values are 2.0
        assert math.isclose(pruned.mean().item(), 1.0, rel_tol=0.1)


class TestSLERP:
    def test_slerp_boundaries(self):
        t1 = torch.tensor([1.0, 0.0])
        t2 = torch.tensor([0.0, 1.0])

        assert (slerp(t1, t2, t=0.0) == t1).all()
        assert (slerp(t1, t2, t=1.0) == t2).all()

        mid = slerp(t1, t2, t=0.5)
        assert math.isclose(mid[0].item(), mid[1].item(), rel_tol=1e-4)
        assert math.isclose(mid.norm().item(), 1.0, rel_tol=1e-4)


class TestFrankensplit:
    def test_layer_splicer(self):
        from nanomind.merge.frankenmerge import LayerSplicer
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])

        m1 = DummyModel()
        m2 = DummyModel()
        spliced = LayerSplicer.splice_layers([m1, m2], [(0, 0, 2), (1, 2, 4)])
        assert len(spliced) == 4


class TestModelMergerAndSoup:
    def test_uniform_soup(self):
        m1 = nn.Linear(2, 2, bias=False)
        m2 = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            m1.weight.fill_(2.0)
            m2.weight.fill_(4.0)
        soup = ModelSoup.uniform_soup([m1, m2])
        assert (soup.weight == 3.0).all()

    def test_model_merger_ties(self):
        base = nn.Linear(4, 2, bias=False)
        ft1 = copy.deepcopy(base)
        ft2 = copy.deepcopy(base)
        with torch.no_grad():
            ft1.weight.add_(torch.randn_like(ft1.weight) * 0.1)
            ft2.weight.add_(torch.randn_like(ft2.weight) * 0.1)

        merger = ModelMerger(MergeConfig(method=MergeMethod.TIES))
        merged = merger.merge(base, [ft1, ft2])
        drift = compute_parameter_drift(base, merged)
        assert drift > 0.0
