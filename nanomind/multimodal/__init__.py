"""NanoMind Multimodal sub-package — Vision + Language fusion.

Implements the full vision-language stack:
  Image → Patches → VisionEncoder → Projector → PrefixFusion → LLM

Follows the LLaVA / InternVL architecture:
  - Patchify images into ViT-style tokens
  - Encode with a lightweight Vision Transformer
  - Project to LLM embedding space via 2-layer MLP
  - Prepend as visual prefix tokens to text input

Primary exports:
    - :class:`ModalityConfig`         — image_size, patch_size, vision_dim, fusion
    - :class:`VisionEncoder`          — ViT-style image encoder
    - :class:`ViTBlock`               — single ViT transformer block
    - :class:`MLPProjector`           — 2-layer MLP vision→LLM projection
    - :class:`PoolingProjector`       — adaptive pool + project
    - :class:`PrefixFusion`           — visual token prefix concatenation
    - :class:`CrossAttentionFusion`   — gated cross-attention (Flamingo-style)
    - :class:`VisionLanguageModel`    — full VLM: encode_image(), n_params
    - :class:`ImageInput`             — single image wrapper
    - :class:`MultimodalInput`        — text + images combined input
    - :class:`MultimodalProcessor`    — encode_images(), build_token_sequence()
    - :func:`patchify`                — image → patch tensor
    - :func:`unpatchify`              — patch tensor → image
    - :class:`ImageNormalizer`        — channel mean/std normalisation
    - :class:`ImageAugmentor`         — crop/flip/color augmentation pipeline
"""

from nanomind.multimodal.config import ModalityConfig
from nanomind.multimodal.patchify import patchify, unpatchify, ImageNormalizer
from nanomind.multimodal.vision_encoder import VisionEncoder, ViTBlock
from nanomind.multimodal.projector import MLPProjector, PoolingProjector
from nanomind.multimodal.fusion import PrefixFusion, CrossAttentionFusion
from nanomind.multimodal.model import VisionLanguageModel
from nanomind.multimodal.inputs import ImageInput, MultimodalInput, MultimodalProcessor
from nanomind.multimodal.augment import (
    ImageAugmentor, RandomCropResize, RandomHorizontalFlip, ColorJitter
)

__all__ = [
    "ModalityConfig",
    "patchify", "unpatchify", "ImageNormalizer",
    "VisionEncoder", "ViTBlock",
    "MLPProjector", "PoolingProjector",
    "PrefixFusion", "CrossAttentionFusion",
    "VisionLanguageModel",
    "ImageInput", "MultimodalInput", "MultimodalProcessor",
    "ImageAugmentor", "RandomCropResize", "RandomHorizontalFlip", "ColorJitter",
]
