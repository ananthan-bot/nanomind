"""
nanomind/flow/model.py — Bidirectional FlowTransformer denoiser with sinusoidal time conditioning.
Allows all token positions to attend to each other simultaneously (non-causal self-attention).
"""
import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.flow.config import FlowConfig


class SinusoidalTimeEmbedding(nn.Module):
    """Projects scalar diffusion timestep t into d_model dimensional embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) scalar timesteps in [0, T] or [0, 1]."""
        half_dim = self.d_model // 2
        emb_scale = math.log(10000.0) / max(1, half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class FlowTransformerBlock(nn.Module):
    """Non-causal bidirectional transformer block with adaptive time conditioning."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # Inject time conditioning via addition
        x = x + time_emb.unsqueeze(1)
        # Bidirectional full attention (no causal triangular mask)
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class FlowTransformer(nn.Module):
    """
    Bidirectional denoiser model predicting clean token distribution across all positions simultaneously.
    """

    def __init__(self, config: Optional[FlowConfig] = None):
        super().__init__()
        self.config = config or FlowConfig()
        d_model = self.config.d_model

        self.token_embed = nn.Embedding(self.config.vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.config.max_seq_len, d_model) * 0.02)
        self.time_embed = SinusoidalTimeEmbedding(d_model)

        self.blocks = nn.ModuleList([
            FlowTransformerBlock(
                d_model=d_model,
                n_heads=self.config.n_heads,
                dim_feedforward=self.config.dim_feedforward,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.config.vocab_size, bias=False)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, T_seq) token indices
        t: (B,) diffusion timestep
        Returns:
            logits: (B, T_seq, vocab_size) clean token predictions
        """
        B, T_seq = x_t.shape
        h = self.token_embed(x_t) + self.pos_embed[:, :T_seq, :]
        t_emb = self.time_embed(t)

        for block in self.blocks:
            h = block(h, t_emb)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits
