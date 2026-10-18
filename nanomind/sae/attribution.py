"""
nanomind/sae/attribution.py — Mechanistic attribution and circuit analysis between SAE latents.
"""
import torch
from typing import Dict, List, Tuple


class FeatureAttribution:
    """
    Traces computational circuits by attributing downstream feature activations
    to upstream SAE dictionary atoms via direct weight interaction matrices.
    """
    @staticmethod
    def compute_interaction_matrix(
        sae_layer1: torch.nn.Module,
        sae_layer2: torch.nn.Module,
        weight_transition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute direct feature-to-feature attribution matrix:
        M = (W_dec1)^T * W_trans * W_enc2^T
        Shape: (d_sae1, d_sae2)
        """
        w_dec1 = sae_layer1.w_dec.weight  # (d_in, d_sae1)
        w_enc2 = sae_layer2.w_enc.weight  # (d_sae2, d_in)

        # Transferred basis: (d_sae1, d_in) @ (d_in, d_in) @ (d_in, d_sae2)
        trans = torch.matmul(w_dec1.t(), weight_transition)
        interaction = torch.matmul(trans, w_enc2.t())
        return interaction

    @staticmethod
    def top_influential_features(
        interaction_matrix: torch.Tensor,
        target_feature_idx: int,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Find the top-k source features that excite the target feature."""
        col = interaction_matrix[:, target_feature_idx]
        vals, indices = torch.topk(col, k=min(top_k, len(col)))
        return [(idx.item(), val.item()) for idx, val in zip(indices, vals)]
