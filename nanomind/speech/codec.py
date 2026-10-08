"""
nanomind/speech/codec.py — Neural Audio Codec: Vector Quantization & Residual Vector Quantization (RVQ).
"""
import math
from typing import Tuple, List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.speech.config import CodecConfig


class VectorQuantizer(nn.Module):
    """
    Standard Vector Quantizer (VQ-VAE style) with Straight-Through Estimator (STE).
    Quantizes continuous latent embeddings into discrete codebook indices.
    """

    def __init__(self, codebook_size: int = 1024, embedding_dim: int = 128, commitment_weight: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight

        # Learnable codebook vectors
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: continuous latents (B, D, T)
        Returns:
            z_q: quantized latents (B, D, T)
            commit_loss: scalar commitment loss
            indices: discrete codes (B, T)
        """
        # Rearrange to (B, T, D)
        z_t = z.permute(0, 2, 1).contiguous()
        B, T, D = z_t.shape

        # Flatten: (B * T, D)
        z_flat = z_t.view(-1, D)

        # Compute squared L2 distance to all codebook vectors: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z @ e^T
        d = (
            torch.sum(z_flat ** 2, dim=-1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=-1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        indices_flat = torch.argmin(d, dim=-1)
        indices = indices_flat.view(B, T)

        # Look up quantized embeddings
        z_q_flat = self.embedding(indices_flat)
        z_q_t = z_q_flat.view(B, T, D)

        # Commitment loss
        loss_commit = F.mse_loss(z_t, z_q_t.detach())
        loss_codebook = F.mse_loss(z_q_t, z_t.detach())
        total_loss = loss_codebook + self.commitment_weight * loss_commit

        # Straight-Through Estimator: copy gradients from z_q to z
        z_q_t = z_t + (z_q_t - z_t).detach()

        # Reshape back to (B, D, T)
        z_q = z_q_t.permute(0, 2, 1).contiguous()
        return z_q, total_loss, indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert discrete indices (B, T) to embeddings (B, D, T)."""
        z_q_t = self.embedding(indices)  # (B, T, D)
        return z_q_t.permute(0, 2, 1).contiguous()
