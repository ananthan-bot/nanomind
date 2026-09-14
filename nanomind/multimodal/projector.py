"""
nanomind/multimodal/projector.py — Visual projector: vision tokens → LLM space.

The projector bridges the vision encoder and the language model.
It maps visual features from the vision embedding space to the LLM
token embedding space so they can be concatenated with text tokens.

LLaVA uses a simple 2-layer MLP:
  vision_token (D_v) → Linear(D_v, D_lm) → GELU → Linear(D_lm, D_lm)

Qwen-VL and InternVL use more complex cross-attention Q-Formers.
NanoMind implements MLP and optional token compression (pooling).

Reference:
  LLaVA: Visual Instruction Tuning — https://arxiv.org/abs/2304.08485
  Q-Former (InstructBLIP): https://arxiv.org/abs/2305.06500
"""

from __future__ import annotations
import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """
    Two-layer MLP projector to map visual features to LLM space.

    Args:
        vision_dim:  Input visual token dimension.
        lm_dim:      Output LLM token dimension.
        hidden_dim:  Hidden layer dimension (default: max of both).

    Example::

        proj   = MLPProjector(vision_dim=128, lm_dim=256)
        visual = torch.randn(1, 196, 128)
        mapped = proj(visual)   # → (1, 196, 256)
    """

    def __init__(
        self,
        vision_dim: int,
        lm_dim:     int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or max(vision_dim, lm_dim)
        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, lm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``(B, N, vision_dim)`` → ``(B, N, lm_dim)``."""
        return self.net(x)


class PoolingProjector(nn.Module):
    """
    Pooling projector: reduce N visual tokens to M tokens via average pooling.

    Reduces sequence length before injecting into the LLM, saving compute.
    Used by efficient VLMs to reduce from 196 → 36 visual tokens.

    Args:
        vision_dim:   Input visual token dimension.
        lm_dim:       Output LLM token dimension.
        target_tokens: Number of output tokens (M < N).
    """

    def __init__(
        self,
        vision_dim:    int,
        lm_dim:        int,
        target_tokens: int = 36,
    ) -> None:
        super().__init__()
        self.target_tokens = target_tokens
        self.proj = nn.Linear(vision_dim, lm_dim)
        self.norm = nn.LayerNorm(lm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool and project ``(B, N, vision_dim)`` → ``(B, target_tokens, lm_dim)``.
        """
        B, N, D = x.shape
        if N > self.target_tokens:
            # Adaptive average pooling over the token dimension
            x = x.permute(0, 2, 1)                         # (B, D, N)
            x = nn.functional.adaptive_avg_pool1d(x, self.target_tokens)
            x = x.permute(0, 2, 1)                         # (B, T, D)
        return self.norm(self.proj(x))


import torch.nn.functional as F   # noqa: E402 (re-import for PoolingProjector)
