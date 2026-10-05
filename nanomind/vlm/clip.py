"""
nanomind/vlm/clip.py — CLIP: Contrastive Language-Image Pre-Training.

## CLIP (Radford et al., 2021)

Train image and text encoders jointly so that:
  - Matching (image, text) pairs → similar embeddings (high cosine sim)
  - Non-matching pairs → dissimilar embeddings (low cosine sim)

## Contrastive Loss (InfoNCE / NT-Xent)

Given batch of N (image, text) pairs:
  - Compute N×N similarity matrix: S[i,j] = cos(img_i, txt_j)
  - Scale by temperature τ (learnable, initialized ~0.07)
  - Cross-entropy loss for image→text direction: predict correct text for each image
  - Cross-entropy loss for text→image direction: predict correct image for each text
  - Total loss = (img→txt + txt→img) / 2

S = img_emb @ txt_emb.T / τ   # (N, N) logits
loss = (CE(S, labels) + CE(S.T, labels)) / 2
labels = [0, 1, 2, ..., N-1]  # each sample is its own positive

## Zero-Shot Image Classification

After CLIP training:
  1. Embed all class names as text: "a photo of a cat", "a photo of a dog"
  2. Embed query image
  3. Find class with highest image-text similarity → zero-shot prediction!

CLIP (ViT-L/14) achieves 76.2% top-1 on ImageNet zero-shot!

## Applications
  - Image search (embed image, search text descriptions)
  - Image captioning (use CLIP features to condition LLM)
  - DALL-E (CLIP guides diffusion model)
  - LLaVA (CLIP vision encoder + LLaMA)

Reference:
  Radford et al. (2021) "Learning Transferable Visual Models From Natural Language Supervision"
  https://arxiv.org/abs/2103.00020
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class CLIPConfig:
    """Configuration for CLIP model."""
    d_embed:        int   = 32     # shared embedding dimension
    temperature_init: float = 0.07  # initial temperature τ
    temperature_max:  float = 100.0
    learn_temperature: bool = True


class CLIPLoss(nn.Module):
    """
    Symmetric InfoNCE contrastive loss for CLIP training.

    Args:
        cfg: :class:`CLIPConfig`.

    Example::

        loss_fn   = CLIPLoss(CLIPConfig())
        img_embs  = F.normalize(torch.randn(8, 32), dim=-1)
        txt_embs  = F.normalize(torch.randn(8, 32), dim=-1)
        loss, acc = loss_fn(img_embs, txt_embs)
        print(f"Loss: {loss.item():.3f}, Accuracy: {acc:.2%}")
    """

    def __init__(self, cfg: CLIPConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CLIPConfig()
        if self.cfg.learn_temperature:
            self.log_temp = nn.Parameter(
                torch.log(torch.tensor(self.cfg.temperature_init))
            )
        else:
            self.register_buffer(
                "log_temp",
                torch.log(torch.tensor(self.cfg.temperature_init)),
            )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(max=self.cfg.temperature_max)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings:  torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Compute symmetric CLIP contrastive loss.

        Args:
            image_embeddings: ``(B, d_embed)`` L2-normalised image vectors.
            text_embeddings:  ``(B, d_embed)`` L2-normalised text vectors.

        Returns:
            Tuple of (loss, retrieval_accuracy).
        """
        B = image_embeddings.shape[0]
        labels = torch.arange(B, device=image_embeddings.device)

        # Similarity matrix: (B, B)
        logits = (image_embeddings @ text_embeddings.T) / self.temperature

        # Symmetric cross-entropy
        loss_i2t = F.cross_entropy(logits,   labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss     = (loss_i2t + loss_t2i) / 2.0

        # Retrieval accuracy: fraction of correct top-1
        with torch.no_grad():
            acc_i2t = (logits.argmax(dim=1) == labels).float().mean()
            acc_t2i = (logits.T.argmax(dim=1) == labels).float().mean()
            acc     = ((acc_i2t + acc_t2i) / 2).item()

        return loss, acc


class CLIPModel(nn.Module):
    """
    Full CLIP model: vision encoder + text encoder + contrastive loss.

    Args:
        vision_encoder: :class:`VisionEncoder`.
        text_encoder:   LM model that returns hidden states.
        d_text:         Text encoder hidden dim.
        cfg:            :class:`CLIPConfig`.

    Example::

        clip   = CLIPModel(vision_enc, text_enc, d_text=64, cfg=CLIPConfig())
        imgs   = torch.randn(4, 3, 32, 32)
        txt    = torch.randint(0, 32, (4, 8))
        loss, acc = clip(imgs, txt)
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder:   nn.Module,
        d_text:         int,
        cfg:            CLIPConfig | None = None,
    ) -> None:
        super().__init__()
        self.vision_enc  = vision_encoder
        self.text_enc    = text_encoder
        self.cfg         = cfg or CLIPConfig()
        self.loss_fn     = CLIPLoss(self.cfg)

        # Text projection to shared embedding space
        self.text_proj = nn.Linear(d_text, self.cfg.d_embed)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to normalised d_embed vectors."""
        cls_emb, _ = self.vision_enc(images)
        return F.normalize(cls_emb, dim=-1)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode token IDs to normalised d_embed vectors."""
        out    = self.text_enc(input_ids)
        hidden = out[0] if isinstance(out, tuple) else out
        # Use last token (or mean pool)
        pooled = hidden.mean(dim=1)          # (B, d_text)
        proj   = self.text_proj(pooled)      # (B, d_embed)
        return F.normalize(proj, dim=-1)

    def forward(
        self,
        images:    torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Forward pass: compute embeddings and CLIP loss.

        Returns:
            (loss, retrieval_accuracy).
        """
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(input_ids)
        return self.loss_fn(img_emb, txt_emb)

    @torch.no_grad()
    def zero_shot_classify(
        self,
        image:       torch.Tensor,
        class_prompts: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Zero-shot image classification.

        Args:
            image:         ``(1, C, H, W)`` image.
            class_prompts: List of token ID tensors (one per class).

        Returns:
            ``(n_classes,)`` softmax probabilities.
        """
        img_emb  = self.encode_image(image)    # (1, d_embed)
        txt_embs = torch.stack([
            self.encode_text(p) for p in class_prompts
        ]).squeeze(1)                          # (n_classes, d_embed)
        logits   = (img_emb @ txt_embs.T) / self.loss_fn.temperature
        return F.softmax(logits[0], dim=-1)

    def image_text_similarity(
        self,
        image:    torch.Tensor,
        text_ids: torch.Tensor,
    ) -> float:
        """Cosine similarity between a single image and text."""
        img = self.encode_image(image.unsqueeze(0))
        txt = self.encode_text(text_ids.unsqueeze(0))
        return (img * txt).sum().item()
