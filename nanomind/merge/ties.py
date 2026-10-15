"""
nanomind/merge/ties.py — TIES-Merging: Trimming, Electing Sign, and Disjoint Merge (Yadav et al., NeurIPS 2023).
Resolves parameter interference across multi-task models.
"""
from typing import List, Dict, Any, Tuple
import torch


class TIESMerger:
    """
    Three-stage TIES merge algorithm:
    1. Trim: Keep top density% largest magnitude parameters, set others to 0.
    2. Elect Sign: Determine consensus sign for each parameter based on total mass.
    3. Disjoint Merge: Average only parameters that agree with the consensus sign.
    """

    def __init__(self, density: float = 0.2):
        self.density = density

    def trim_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Retain top density% parameters by absolute magnitude."""
        if self.density >= 1.0:
            return t
        flat = t.abs().flatten()
        k = max(1, int(self.density * flat.numel()))
        threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
        mask = t.abs() >= threshold
        return t * mask

    def merge_tensors(self, tensors: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        Executes TIES on a list of delta tensors.
        """
        n = len(tensors)
        w = weights or [1.0 / n] * n

        # 1. Trim
        trimmed = [self.trim_tensor(t) for t in tensors]

        # 2. Elect Sign
        stacked = torch.stack(trimmed, dim=0)  # (N, ...)
        sign_vote = torch.zeros_like(stacked[0])
        for i in range(n):
            sign_vote += w[i] * torch.sign(stacked[i])
        consensus_sign = torch.sign(sign_vote)

        # 3. Disjoint Merge
        merged = torch.zeros_like(stacked[0])
        denom = torch.zeros_like(stacked[0])

        for i in range(n):
            agrees = (torch.sign(stacked[i]) == consensus_sign) & (stacked[i] != 0)
            merged += w[i] * (stacked[i] * agrees)
            denom += w[i] * agrees.float()

        denom = torch.clamp(denom, min=1e-8)
        return merged / denom
