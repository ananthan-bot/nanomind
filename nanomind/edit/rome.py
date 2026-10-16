"""
nanomind/edit/rome.py — Rank-One Model Editing (ROME, Meng et al., NeurIPS 2022).
Closed-form rank-one weight update: Delta W = (v* - W k*) (k*^T C^(-1)) / (k*^T C^(-1) k*).
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from nanomind.edit.config import EditConfig


class ROMEEngine:
    """
    Implements closed-form Rank-One Model Editing (ROME) on Linear projection matrices.
    Guarantees: (W + Delta W) k* = v* while minimizing perturbation on unedited keys.
    """

    def __init__(self, config: Optional[EditConfig] = None):
        self.config = config or EditConfig()

    def compute_weight_update(
        self,
        weight: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        c_inv: torch.Tensor,
    ) -> torch.Tensor:
        """
        weight: (D_out, D_in) target weight matrix W
        key: (D_in,) key representation k*
        value: (D_out,) target value representation v*
        c_inv: (D_in, D_in) inverse covariance matrix C^(-1)
        Returns:
            delta_w: (D_out, D_in) rank-one weight update
        """
        # Residual error vector: e = v* - W k*
        residual = value - torch.matmul(weight, key)  # (D_out,)

        # Project key via C^(-1)
        c_inv_k = torch.matmul(c_inv, key)  # (D_in,)
        denom = torch.dot(key, c_inv_k)      # scalar k*^T C^(-1) k*
        denom = torch.clamp(denom, min=1e-8)

        # Rank-one outer product: (e) (C^(-1) k*)^T / denom
        delta_w = torch.outer(residual, c_inv_k) / denom
        return delta_w

    def apply_edit(
        self,
        layer_module: nn.Linear,
        key: torch.Tensor,
        value: torch.Tensor,
        c_inv: torch.Tensor,
    ):
        """Applies rank-one update directly to nn.Linear layer."""
        with torch.no_grad():
            delta_w = self.compute_weight_update(layer_module.weight.data, key, value, c_inv)
            layer_module.weight.data.add_(delta_w)
