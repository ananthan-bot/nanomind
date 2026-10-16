"""
nanomind/edit/kv_extract.py — Key-Value representation extraction at Transformer MLP layers.
Extracts intermediate key k* at subject tokens and optimizes target value v*.
"""
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn


class KeyValueExtractor:
    """
    Extracts key vector k* = act(W_in * x) representing the subject,
    and optimizes value vector v* satisfying W_out * k* = v*.
    """

    @staticmethod
    def extract_key(mlp_in_activation: torch.Tensor) -> torch.Tensor:
        """
        mlp_in_activation: (D_in,) hidden state representation at subject's final token.
        Returns normalized key vector k*.
        """
        k = mlp_in_activation.clone().detach()
        return k / (torch.norm(k) + 1e-8)

    @staticmethod
    def compute_target_value(
        current_weight: torch.Tensor,
        key: torch.Tensor,
        target_embedding: torch.Tensor,
        blend_factor: float = 0.8,
    ) -> torch.Tensor:
        """
        Computes ideal target value vector v*:
        v* = blend_factor * target_embedding + (1 - blend_factor) * (current_weight @ key)
        """
        current_v = torch.matmul(current_weight, key)
        v_star = blend_factor * target_embedding + (1.0 - blend_factor) * current_v
        return v_star
