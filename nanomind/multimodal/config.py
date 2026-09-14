"""
nanomind/multimodal/config.py — Multimodal model configuration.

## How Multimodal LLMs Work

Vision-Language Models (VLMs) extend text LLMs to understand images:

  1. Image Patching (ViT-style):
       Input image (H × W × 3) → divide into patches (P × P pixels each)
       → N = (H/P) × (W/P) patches
       → Linear projection → N "image tokens" of dimension D

  2. Fusion (Early vs Late):
       Early: concatenate image tokens with text tokens before attention
              → "Prefix" style: [IMG_1, IMG_2, ..., IMG_N, text tokens]
              Used by: LLaVA, InternVL, Qwen-VL

       Late:  separate vision and text encoders, cross-attention to fuse
              Used by: Flamingo, CogVLM

  3. Special tokens:
       <image>    → placeholder in text prompt
       <img_start> / <img_end> → delimit image tokens in sequence

Models implementing this:
  LLaVA (Haotian Liu, 2023):   CLIP vision + Vicuna LLM
  InstructBLIP (Salesforce):   Q-Former for image-text alignment
  GPT-4V (OpenAI, 2023):       Proprietary
  Gemini (Google, 2023):       Native multimodal
  Qwen-VL (Alibaba, 2023):    Compact open VLM

References:
  LLaVA: https://arxiv.org/abs/2304.08485
  Flamingo: https://arxiv.org/abs/2204.14198
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ModalityConfig:
    """
    Configuration for multimodal (vision-language) model.

    Attributes:
        image_size:     Input image size (square, pixels).
        patch_size:     Patch size (must divide image_size evenly).
        vision_dim:     Vision encoder output dimension.
        fusion:         Fusion strategy: ``"prefix"`` or ``"cross"``.
        n_visual_tokens: Number of visual tokens after projection.
        image_token_id: Special token ID for ``<image>`` placeholder.
        img_start_id:   Token ID for ``<img_start>`` marker.
        img_end_id:     Token ID for ``<img_end>`` marker.
        normalize_mean: Per-channel mean for image normalisation.
        normalize_std:  Per-channel std for image normalisation.
    """
    image_size:      int       = 224
    patch_size:      int       = 16
    vision_dim:      int       = 128
    fusion:          str       = "prefix"
    n_visual_tokens: int       = 49         # (224/16)^2 = 196 → projected to 49
    image_token_id:  int       = 0
    img_start_id:    int       = 1
    img_end_id:      int       = 2
    normalize_mean:  list      = None
    normalize_std:   list      = None

    def __post_init__(self) -> None:
        if self.normalize_mean is None:
            self.normalize_mean = [0.485, 0.456, 0.406]   # ImageNet
        if self.normalize_std is None:
            self.normalize_std  = [0.229, 0.224, 0.225]
        assert self.image_size  % self.patch_size == 0
        assert self.fusion      in ("prefix", "cross")
        assert self.patch_size  >= 4
        assert self.vision_dim  >= 8

    @property
    def n_patches(self) -> int:
        """Total number of patches per image."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        """Flattened patch dimension (C × P × P)."""
        return 3 * self.patch_size * self.patch_size
