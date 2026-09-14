"""
nanomind/multimodal/inputs.py — Multimodal input types and processors.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class ImageInput:
    """
    A single image input for multimodal inference.

    Attributes:
        pixels:   Image tensor ``(C, H, W)`` in [0, 1] float32.
        metadata: Optional dict with source path, alt text, etc.
    """
    pixels:   torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        return tuple(self.pixels.shape)

    @classmethod
    def zeros(cls, size: int = 224) -> "ImageInput":
        """Create a blank (zero) image of given size."""
        return cls(pixels=torch.zeros(3, size, size))

    @classmethod
    def random(cls, size: int = 224) -> "ImageInput":
        """Create a random image for testing."""
        return cls(pixels=torch.rand(3, size, size))


@dataclass
class MultimodalInput:
    """
    Combined text + image input for a multimodal model.

    Attributes:
        text:   Text prompt (may contain ``<image>`` placeholder).
        images: List of :class:`ImageInput` objects.
        system: Optional system message.
    """
    text:   str
    images: list[ImageInput] = field(default_factory=list)
    system: str | None       = None

    @property
    def n_images(self) -> int:
        return len(self.images)

    def has_images(self) -> bool:
        return len(self.images) > 0

    def image_placeholder_count(self) -> int:
        """Count <image> placeholders in the text."""
        return self.text.count("<image>")


class MultimodalProcessor:
    """
    Process multimodal inputs: encode images and build token sequences.

    Args:
        encoder:   :class:`VisionEncoder` for image → visual tokens.
        projector: :class:`MLPProjector` for visual token → LLM space.
        tokenizer: Text tokenizer with encode/decode.
        cfg:       :class:`ModalityConfig`.
    """

    def __init__(self, encoder, projector, tokenizer, cfg) -> None:
        self.encoder   = encoder
        self.projector = projector
        self.tokenizer = tokenizer
        self.cfg       = cfg

    @torch.no_grad()
    def encode_images(self, images: list[ImageInput]) -> torch.Tensor:
        """
        Encode a list of images to projected visual tokens.

        Args:
            images: List of :class:`ImageInput`.

        Returns:
            ``(N_images, N_patches, lm_dim)`` visual token tensor.
        """
        pixels = torch.stack([img.pixels for img in images])  # (N, C, H, W)
        visual = self.encoder(pixels)                           # (N, N_patches, vision_dim)
        return self.projector(visual)                           # (N, N_patches, lm_dim)

    def build_token_sequence(
        self,
        mm_input:    "MultimodalInput",
        visual_embeds: torch.Tensor | None = None,
    ) -> dict:
        """
        Build the combined token sequence for multimodal input.

        Returns:
            Dict with ``input_ids``, ``image_positions``, ``n_visual_tokens``.
        """
        text_ids = self.tokenizer.encode(mm_input.text)
        return {
            "input_ids":       text_ids,
            "n_images":        mm_input.n_images,
            "n_visual_tokens": self.cfg.n_visual_tokens,
        }
