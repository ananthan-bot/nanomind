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


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ) as used in SoundStream and EnCodec.
    Cascades n_q quantizers where each quantizer models the residual of previous stages.
    """

    def __init__(self, config: Optional[CodecConfig] = None):
        super().__init__()
        self.config = config or CodecConfig()
        self.n_q = self.config.n_q
        self.codebook_size = self.config.codebook_size
        self.embedding_dim = self.config.embedding_dim

        # n_q quantizer stages
        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                codebook_size=self.codebook_size,
                embedding_dim=self.embedding_dim,
                commitment_weight=self.config.commitment_weight,
            )
            for _ in range(self.n_q)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: continuous audio latents (B, D, T)
        Returns:
            z_q_total: aggregated quantized latents (B, D, T)
            total_loss: summed commitment/codebook loss across stages
            all_indices: multi-stage discrete codes (B, n_q, T)
        """
        B, D, T = z.shape
        residual = z
        z_q_total = torch.zeros_like(z)
        total_loss = torch.tensor(0.0, device=z.device)
        indices_list = []

        for vq in self.quantizers:
            z_q_stage, loss_stage, idx_stage = vq(residual)
            residual = residual - z_q_stage
            z_q_total = z_q_total + z_q_stage
            total_loss = total_loss + loss_stage
            indices_list.append(idx_stage.unsqueeze(1))  # (B, 1, T)

        all_indices = torch.cat(indices_list, dim=1)  # (B, n_q, T)
        return z_q_total, total_loss, all_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct audio latents from multi-stage codes.
        codes: (B, n_q, T)
        Returns: (B, D, T) reconstructed latents
        """
        B, n_q, T = codes.shape
        z_out = torch.zeros(B, self.embedding_dim, T, device=codes.device)
        for i in range(min(n_q, self.n_q)):
            vq = self.quantizers[i]
            z_stage = vq.decode_indices(codes[:, i, :])
            z_out = z_out + z_stage
        return z_out
