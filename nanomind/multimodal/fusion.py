"""
nanomind/multimodal/fusion.py — Image-text fusion strategies.

Defines how visual tokens are merged with text token embeddings:

  Prefix Fusion (LLaVA-style):
    [<img_start>, v_1, v_2, ..., v_N, <img_end>, text tokens...]
    Simple concatenation — visual tokens prefix the text.
    The LLM sees N additional "virtual" tokens before the text.

  Cross-Attention Fusion (Flamingo-style):
    Text tokens attend to visual tokens via gated cross-attention.
    More parameter-efficient but architecturally more complex.
    Allows the LLM to selectively query visual information.

NanoMind implements prefix fusion (simpler, more widely used).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class PrefixFusion(nn.Module):
    """
    Prefix fusion: prepend projected visual tokens to text embeddings.

    Args:
        lm_embed_dim: LLM embedding dimension.

    Example::

        fusion  = PrefixFusion(lm_embed_dim=256)
        vis_tok = torch.randn(1, 49, 256)   # projected visual tokens
        txt_emb = torch.randn(1, 20, 256)   # text token embeddings
        fused   = fusion(vis_tok, txt_emb)
        # → (1, 69, 256)   visual prefix + text
    """

    def __init__(self, lm_embed_dim: int) -> None:
        super().__init__()
        self.lm_embed_dim = lm_embed_dim
        # Learnable scale for visual tokens (initialised to 1)
        self.visual_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        visual_embeds: torch.Tensor,
        text_embeds:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse visual prefix with text embeddings.

        Args:
            visual_embeds: ``(B, N_visual, D)`` projected visual tokens.
            text_embeds:   ``(B, N_text, D)`` text token embeddings.

        Returns:
            ``(B, N_visual + N_text, D)`` fused embedding sequence.
        """
        scaled_vis = visual_embeds * self.visual_scale
        return torch.cat([scaled_vis, text_embeds], dim=1)

    @property
    def output_len(self) -> str:
        return "N_visual + N_text"


class CrossAttentionFusion(nn.Module):
    """
    Gated cross-attention fusion (Flamingo-style).

    Text tokens attend to visual tokens via cross-attention.
    A tanh-gated residual controls the visual influence.

    Args:
        lm_embed_dim: LLM token dimension.
        n_heads:      Number of cross-attention heads.
    """

    def __init__(self, lm_embed_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            lm_embed_dim, n_heads, batch_first=True
        )
        self.norm  = nn.LayerNorm(lm_embed_dim)
        self.gate  = nn.Parameter(torch.zeros(1))   # tanh gate, starts at 0

    def forward(
        self,
        visual_embeds: torch.Tensor,
        text_embeds:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-attend text tokens to visual tokens.

        Args:
            visual_embeds: ``(B, N_visual, D)`` visual context.
            text_embeds:   ``(B, N_text, D)`` text queries.

        Returns:
            ``(B, N_text, D)`` text tokens with injected visual context.
        """
        normed  = self.norm(text_embeds)
        attn_out, _ = self.cross_attn(
            normed, visual_embeds, visual_embeds,
            need_weights=False,
        )
        # Gated residual: tanh(gate) starts at 0, learns to open gradually
        return text_embeds + torch.tanh(self.gate) * attn_out
