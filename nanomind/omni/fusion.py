"""
nanomind/omni/fusion.py — Modality type embeddings and cross-modal fusion attention.
"""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.omni.config import ModalityType


class ModalityEmbedding(nn.Module):
    """
    Learned embedding distinguishing modality types (text=0, image=1, audio=2, tool=3).
    Added to representations prior to transformer processing.
    """

    def __init__(self, num_modalities: int = 4, d_model: int = 256):
        super().__init__()
        self.embed = nn.Embedding(num_modalities, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, modality_ids: torch.Tensor) -> torch.Tensor:
        """
        modality_ids: (B, T)
        Returns: (B, T, D)
        """
        return self.embed(modality_ids)


class CrossModalFusionLayer(nn.Module):
    """
    Modality-aware transformer layer with gated Pre-LayerNorm residual connections.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN Self-Attention
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=mask)
        x = x + attn_out

        # Pre-LN FFN
        x = x + self.ffn(self.ln2(x))
        return x
