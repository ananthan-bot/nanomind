"""
nanomind/edit/memit.py — Mass-Editing Memory in a Transformer (MEMIT, Meng et al., ICLR 2023).
Batch edits thousands of factual associations distributed across multiple Transformer layers.
"""
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn

from nanomind.edit.config import EditConfig


class MEMITEngine:
    """
    Mass-Editing Memory in a Transformer (MEMIT).
    Solves for joint weight update: Delta W = (V - W K) (C + K K^T)^(-1) K^T,
    distributing residual values across a range of consecutive layers.
    """

    def __init__(self, config: Optional[EditConfig] = None):
        self.config = config or EditConfig(algorithm="memit")

    def compute_batch_update(
        self,
        weight: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        c_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        weight: (D_out, D_in)
        keys: (D_in, M) batch of M keys
        values: (D_out, M) batch of M target values
        c_matrix: (D_in, D_in) covariance matrix C
        Returns:
            delta_w: (D_out, D_in)
        """
        D_in, M = keys.shape
        D_out = values.shape[0]

        # Residuals across all M facts: R = V - W K of shape (D_out, M)
        residuals = values - torch.matmul(weight, keys)

        # Regulerized joint inverse: (C + K K^T)^(-1)
        kk_t = torch.matmul(keys, keys.t())
        joint_c = c_matrix + kk_t + torch.eye(D_in, device=c_matrix.device) * self.config.ridge_lambda
        joint_inv = torch.linalg.inv(joint_c)

        # Delta W = R K^T joint_inv
        delta_w = torch.matmul(residuals, torch.matmul(keys.t(), joint_inv))
        return delta_w
