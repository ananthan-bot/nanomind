"""
nanomind/multimodal/patchify.py — Image patching (ViT-style).

Splits an image into non-overlapping patches, the standard approach
used by Vision Transformers (ViT, DeiT, CLIP, etc.).

Pipeline:
  Image (H, W, C)                      → raw pixels
  Normalise (subtract mean, divide std) → float32 in ~[-2, 2]
  Reshape into patches (N, P*P*C)       → N = (H/P)*(W/P) patches
  Linear projection (N, D)              → visual tokens for the LLM

Reference:
  Dosovitskiy et al. (2020) "An Image is Worth 16x16 Words"
  https://arxiv.org/abs/2010.11929
"""

from __future__ import annotations
import torch
import torch.nn as nn


def patchify(
    image:      torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """
    Split a batch of images into flattened patches.

    Args:
        image:      ``(B, C, H, W)`` image tensor.
        patch_size: Patch side length in pixels.

    Returns:
        ``(B, N, C*P*P)`` patch tensor where N = (H/P)*(W/P).

    Example::

        img     = torch.randn(1, 3, 224, 224)
        patches = patchify(img, patch_size=16)
        # → (1, 196, 768)   (196 patches, each 3×16×16=768-dim)
    """
    B, C, H, W = image.shape
    P           = patch_size
    assert H % P == 0 and W % P == 0,         f"Image size ({H}×{W}) must be divisible by patch_size ({P})"

    # Reshape: B, C, H/P, P, W/P, P  →  B, N, C*P*P
    x = image.reshape(B, C, H // P, P, W // P, P)
    x = x.permute(0, 2, 4, 1, 3, 5)          # B, H/P, W/P, C, P, P
    x = x.contiguous().reshape(B, -1, C * P * P)
    return x


def unpatchify(
    patches:    torch.Tensor,
    patch_size: int,
    image_size: int,
    channels:   int = 3,
) -> torch.Tensor:
    """
    Reconstruct images from patches (inverse of patchify).

    Args:
        patches:    ``(B, N, C*P*P)`` patch tensor.
        patch_size: Patch side length.
        image_size: Output image side length.
        channels:   Number of image channels.

    Returns:
        ``(B, C, H, W)`` image tensor.
    """
    B   = patches.shape[0]
    P   = patch_size
    G   = image_size // P
    C   = channels
    x   = patches.reshape(B, G, G, C, P, P)
    x   = x.permute(0, 3, 1, 4, 2, 5)        # B, C, G, P, G, P
    return x.contiguous().reshape(B, C, image_size, image_size)


class ImageNormalizer(nn.Module):
    """
    Normalise image tensors using channel-wise mean and std.

    Args:
        mean: Per-channel mean (default: ImageNet mean).
        std:  Per-channel std (default: ImageNet std).
    """

    def __init__(
        self,
        mean: list[float] = None,
        std:  list[float] = None,
    ) -> None:
        super().__init__()
        mean = mean or [0.485, 0.456, 0.406]
        std  = std  or [0.229, 0.224, 0.225]
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``(B, 3, H, W)`` image tensor."""
        return (x - self.mean) / (self.std + 1e-7)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse normalisation."""
        return x * self.std + self.mean
