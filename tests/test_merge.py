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
