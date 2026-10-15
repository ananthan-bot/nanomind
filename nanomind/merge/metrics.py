"""
nanomind/merge/metrics.py — Metrics for parameter drift, task interference, and sparsity.
"""
from typing import Dict, Any, List
import math
import torch
import torch.nn as nn


def compute_parameter_drift(base_model: nn.Module, merged_model: nn.Module) -> float:
    """
    Computes Euclidean L2 parameter drift ||theta_merged - theta_base||_2.
    """
    total_sq = 0.0
    base_params = dict(base_model.named_parameters())
    merged_params = dict(merged_model.named_parameters())

    for name in base_params:
        if name in merged_params:
            diff = merged_params[name].data - base_params[name].data
            total_sq += torch.sum(diff ** 2).item()

    return math.sqrt(total_sq)


def compute_task_vector_cosine_similarity(tv1: Dict[str, torch.Tensor], tv2: Dict[str, torch.Tensor]) -> float:
    """
    Measures cosine similarity between two task vectors.
    High negative similarity indicates conflicting task directions.
    """
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for name in tv1:
        if name in tv2:
            t1 = tv1[name].flatten()
            t2 = tv2[name].flatten()
            dot += torch.sum(t1 * t2).item()
            norm1 += torch.sum(t1 ** 2).item()
            norm2 += torch.sum(t2 ** 2).item()

    denom = math.sqrt(norm1) * math.sqrt(norm2)
    return float(dot / max(1e-8, denom))
