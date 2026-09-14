"""
nanomind/multimodal/augment.py — Image augmentation for vision-language training.

Augmentation is critical for robust visual grounding:
  - Random crop + resize: model learns scale invariance
  - Color jitter: model learns color-invariant features
  - Random flip:  model learns spatial invariance (for non-directional tasks)

Note: Some VQA tasks are directional (left/right), so horizontal flipping
must be disabled for position-sensitive tasks.

References:
  SimCLR augmentation: https://arxiv.org/abs/2002.05709
  CLIP training: https://arxiv.org/abs/2103.00020
"""

from __future__ import annotations
import random
import torch
import torch.nn.functional as F


class RandomCropResize:
    """Random crop and resize to target size."""

    def __init__(self, size: int = 224, scale: tuple = (0.8, 1.0)) -> None:
        self.size  = size
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply to ``(C, H, W)`` tensor."""
        C, H, W = x.shape
        scale   = random.uniform(*self.scale)
        new_h   = int(H * scale)
        new_w   = int(W * scale)
        top     = random.randint(0, H - new_h)
        left    = random.randint(0, W - new_w)
        x       = x[:, top:top + new_h, left:left + new_w]
        x       = F.interpolate(x.unsqueeze(0), size=(self.size, self.size),
                                mode="bilinear", align_corners=False).squeeze(0)
        return x


class RandomHorizontalFlip:
    """Randomly flip image horizontally with given probability."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return x.flip(-1)
        return x


class ColorJitter:
    """Random brightness, contrast, saturation, hue jitter."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast:   float = 0.2,
        saturation: float = 0.1,
    ) -> None:
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply to ``(C, H, W)`` tensor in [0, 1]."""
        # Brightness
        if self.brightness > 0:
            f = 1.0 + random.uniform(-self.brightness, self.brightness)
            x = (x * f).clamp(0.0, 1.0)
        # Contrast
        if self.contrast > 0:
            mean = x.mean(dim=(-2, -1), keepdim=True)
            f    = 1.0 + random.uniform(-self.contrast, self.contrast)
            x    = ((x - mean) * f + mean).clamp(0.0, 1.0)
        return x


class ImageAugmentor:
    """
    Compose image augmentations for vision-language training.

    Args:
        image_size:  Target image size after augmentation.
        flip:        Include random horizontal flip.
        crop:        Include random crop-resize.
        color:       Include color jitter.

    Example::

        aug   = ImageAugmentor(image_size=224)
        pixel = torch.rand(3, 256, 256)
        out   = aug(pixel)   # → (3, 224, 224)
    """

    def __init__(
        self,
        image_size: int   = 224,
        flip:       bool  = True,
        crop:       bool  = True,
        color:      bool  = True,
    ) -> None:
        self.transforms = []
        if crop:
            self.transforms.append(RandomCropResize(image_size))
        if flip:
            self.transforms.append(RandomHorizontalFlip())
        if color:
            self.transforms.append(ColorJitter())
        self._resize = image_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        # Ensure correct output size
        if x.shape[-1] != self._resize or x.shape[-2] != self._resize:
            x = F.interpolate(x.unsqueeze(0), size=(self._resize, self._resize),
                              mode="bilinear", align_corners=False).squeeze(0)
        return x
