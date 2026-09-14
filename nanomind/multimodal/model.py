"""
nanomind/multimodal/model.py — Full vision-language model.

Combines:
  VisionEncoder → MLPProjector → PrefixFusion → NanoMind LLM
"""

from __future__ import annotations
import torch
import torch.nn as nn

from nanomind.multimodal.config import ModalityConfig
from nanomind.multimodal.vision_encoder import VisionEncoder
from nanomind.multimodal.projector import MLPProjector, PoolingProjector
from nanomind.multimodal.fusion import PrefixFusion, CrossAttentionFusion
from nanomind.multimodal.inputs import MultimodalInput, ImageInput
from nanomind.utils.logger import get_logger

log = get_logger("multimodal.model")


class VisionLanguageModel(nn.Module):
    """
    Vision-Language Model: encode images and generate text.

    Combines a VisionEncoder, a projector, and any text LLM backbone.

    Args:
        lm:          Language model backbone (must have ``lm.tok_emb``).
        cfg:         Modality configuration.
        lm_dim:      LLM embedding dimension.
        n_vis_layers: Number of ViT layers in the vision encoder.

    Example::

        vlm   = VisionLanguageModel(lm=nanomind_model, cfg=ModalityConfig(), lm_dim=128)
        image = ImageInput.random()
        out   = vlm.encode_image(image)   # → (1, N, lm_dim) visual tokens
    """

    def __init__(
        self,
        lm:           nn.Module,
        cfg:          ModalityConfig,
        lm_dim:       int,
        n_vis_layers: int = 2,
        n_vis_heads:  int = 4,
        use_pooling:  bool = True,
    ) -> None:
        super().__init__()
        self.cfg     = cfg
        self.lm      = lm
        self.encoder = VisionEncoder(cfg, n_layers=n_vis_layers, n_heads=n_vis_heads)
        self.projector = (
            PoolingProjector(cfg.vision_dim, lm_dim, target_tokens=cfg.n_visual_tokens)
            if use_pooling
            else MLPProjector(cfg.vision_dim, lm_dim)
        )
        self.fusion  = (
            PrefixFusion(lm_dim)
            if cfg.fusion == "prefix"
            else CrossAttentionFusion(lm_dim)
        )

    @torch.no_grad()
    def encode_image(self, image: "ImageInput") -> torch.Tensor:
        """
        Encode a single image to projected visual tokens.

        Args:
            image: :class:`ImageInput`.

        Returns:
            ``(1, N_visual, lm_dim)`` tensor.
        """
        pixels  = image.pixels.unsqueeze(0)               # (1, C, H, W)
        visual  = self.encoder(pixels)                     # (1, N_patch, vision_dim)
        return self.projector(visual)                      # (1, N_visual, lm_dim)

    def visual_token_count(self) -> int:
        """Number of visual tokens produced per image."""
        return self.cfg.n_visual_tokens

    @property
    def n_params(self) -> dict:
        """Parameter counts by component."""
        return {
            "vision_encoder": self.encoder.n_params,
            "projector":      sum(p.numel() for p in self.projector.parameters()),
            "fusion":         sum(p.numel() for p in self.fusion.parameters()),
            "lm":             sum(p.numel() for p in self.lm.parameters()),
            "total":          sum(p.numel() for p in self.parameters()),
        }
