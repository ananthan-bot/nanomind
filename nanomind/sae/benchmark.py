"""
nanomind/sae/benchmark.py — Comprehensive evaluation metrics for Sparse Autoencoders.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Any


class SAEBenchmark:
    """
    Evaluates quality of dictionary representations:
    1. Normalized Mean Squared Error (NMSE): ||x - x_hat||^2 / ||x||^2
    2. Explained Variance: 1 - Var(x - x_hat) / Var(x)
    3. L0 Sparsity: Mean count of active features
    4. Dead Feature Ratio: Fraction of never-activated dictionary atoms
    """
    @staticmethod
    def evaluate(
        sae: torch.nn.Module,
        test_activations: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute full benchmark suite over test activations."""
        with torch.no_grad():
            out = sae(test_activations)
            x = test_activations
            x_hat = out.reconstruction

            # NMSE
            recon_err = (x - x_hat).pow(2).sum()
            x_norm = x.pow(2).sum().clamp(min=1e-8)
            nmse = (recon_err / x_norm).item()

            # Explained Variance
            var_err = (x - x_hat).var()
            var_x = x.var().clamp(min=1e-8)
            explained_var = max(0.0, (1.0 - var_err / var_x).item())

            # L0
            l0 = (out.latents > 0).float().sum(dim=-1).mean().item()

            # Dead feature ratio
            active_cols = (out.latents.sum(dim=0) > 0).float()
            dead_ratio = 1.0 - active_cols.mean().item()

        return {
            "nmse": round(nmse, 4),
            "explained_variance": round(explained_var, 4),
            "l0_sparsity": round(l0, 2),
            "dead_feature_ratio": round(dead_ratio, 4),
            "passed_reconstruction": explained_var >= 0.80,
        }
