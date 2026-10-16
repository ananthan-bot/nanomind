"""
nanomind/edit/covariance.py — Covariance matrix collection C = E[k k^T] and regularized inversion.
Preserves unedited facts by projecting weight updates onto the null space of unedited keys.
"""
from typing import Optional
import torch


class CovarianceCollector:
    """
    Maintains second-moment matrix C = 1/N sum_i k_i k_i^T over diverse factual prompts.
    """

    def __init__(self, d_model: int, ridge_lambda: float = 1e-4):
        self.d_model = d_model
        self.ridge_lambda = ridge_lambda
        self.cov_matrix = torch.eye(d_model) * ridge_lambda
        self.sample_count = 0

    def update(self, key_batch: torch.Tensor):
        """
        key_batch: (N, d_model) keys from random Wikipedia or pretraining samples.
        """
        N, D = key_batch.shape
        batch_cov = torch.matmul(key_batch.t(), key_batch) / max(1, N)
        self.cov_matrix = (self.cov_matrix * self.sample_count + batch_cov * N) / (self.sample_count + N)
        self.sample_count += N

    def get_inverse(self) -> torch.Tensor:
        """Returns regularized inverse (C + lambda * I)^(-1)."""
        reg_c = self.cov_matrix + torch.eye(self.d_model, device=self.cov_matrix.device) * self.ridge_lambda
        return torch.linalg.inv(reg_c)
