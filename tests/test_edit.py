"""
tests/test_edit.py — Comprehensive unit tests for Model Editing algorithms (ROME / MEMIT).
"""
import math
import torch
import torch.nn as nn
from nanomind.edit import (
    KnowledgeFact,
    EditConfig,
    CausalTracer,
    KeyValueExtractor,
    CovarianceCollector,
    ROMEEngine,
    MEMITEngine,
    ModelEditor,
    EditEvaluator,
)


class TestKnowledgeFact:
    def test_fact_creation(self):
        fact = KnowledgeFact(
            prompt="The president of USA is",
            subject="USA",
            target_new="Alice",
            target_true="Bob",
            paraphrases=["The current leader of USA is"],
        )
        assert fact.subject == "USA"
        assert fact.target_new == "Alice"
        assert len(fact.paraphrases) == 1


class TestCausalTracing:
    def test_corrupt_embeddings(self):
        tracer = CausalTracer(noise_std=0.2)
        emb = torch.zeros(1, 4, 16)
        corrupted = tracer.corrupt_embeddings(emb, token_indices=[1, 2])
        # Indices 0 and 3 should remain 0
        assert (corrupted[:, 0, :] == 0).all()
        assert (corrupted[:, 3, :] == 0).all()
        # Indices 1 and 2 should have noise
        assert (corrupted[:, 1, :] != 0).any()

    def test_indirect_effect_metric(self):
        tracer = CausalTracer()
        # Full restoration: clean=1.0, corrupted=0.0, restored=1.0 -> 1.0
        ie = tracer.compute_indirect_effect(1.0, 0.0, 1.0)
        assert math.isclose(ie, 1.0, abs_tol=1e-5)

        # Zero restoration: clean=1.0, corrupted=0.0, restored=0.0 -> 0.0
        ie_zero = tracer.compute_indirect_effect(1.0, 0.0, 0.0)
        assert math.isclose(ie_zero, 0.0, abs_tol=1e-5)


class TestCovariance:
    def test_covariance_inversion(self):
        cov = CovarianceCollector(d_model=8, ridge_lambda=0.01)
        samples = torch.randn(50, 8)
        cov.update(samples)
        inv = cov.get_inverse()
        assert inv.shape == (8, 8)
        # Check product with regularized C is approximately identity
        reg_c = cov.cov_matrix + torch.eye(8) * 0.01
        prod = torch.matmul(reg_c, inv)
        eye = torch.eye(8)
        assert torch.allclose(prod, eye, atol=1e-4)


class TestROME:
    def test_exact_interpolation(self):
        rome = ROMEEngine()
        d_in, d_out = 16, 16
        w = torch.randn(d_out, d_in)
        k = torch.randn(d_in)
        k = k / k.norm()
        v = torch.randn(d_out)
        c_inv = torch.eye(d_in)

        delta_w = rome.compute_weight_update(w, k, v, c_inv)
        w_new = w + delta_w
        # (w + delta_w) k must equal v exactly!
        v_pred = torch.matmul(w_new, k)
        assert torch.allclose(v_pred, v, atol=1e-5)
