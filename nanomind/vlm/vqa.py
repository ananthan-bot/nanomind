"""
nanomind/vlm/vqa.py — Visual Question Answering (VQA).

## VQA Task

Given: image + natural language question
Produce: natural language answer

VQA datasets:
  - VQA v2: 1.1M questions about COCO images
  - GQA: compositional questions requiring spatial/logical reasoning
  - TextVQA: questions requiring reading text in images
  - ScienceQA: multimodal science questions with explanations

## VQA Model Architectures

1. Early Fusion (LLaVA style):
   visual_tokens = ViT(image) → projector
   input = [visual_tokens] + tokenize(question)
   answer = LLM(input)

2. Encoder-Decoder:
   visual_features = ViT(image)
   encoder_output  = cross_attn(text_features, visual_features)
   answer          = decoder(encoder_output)

3. BLIP-2 (Li et al., 2023):
   image → ViT → Q-Former (32 learnable queries attend to patches) → LLM
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from nanomind.vlm.cross_attention import VisualProjector, CrossModalAttention, CrossAttentionConfig


@dataclass
class VQAConfig:
    """Configuration for the VQA model."""
    d_visual:    int   = 32    # vision encoder output dim
    d_model:     int   = 64    # LM hidden dim
    vocab_size:  int   = 256
    max_answer_len: int = 20
    fusion:      str   = "early"   # "early" | "cross_attention"


@dataclass
class VQASample:
    """A single VQA example."""
    image_id:  str
    question:  str
    answer:    str
    q_type:    str = "what"   # what/where/who/how many/yes-no

    def to_dict(self) -> dict:
        return {"image_id": self.image_id, "question": self.question,
                "answer":   self.answer, "q_type": self.q_type}


@dataclass
class VQAResult:
    """Model prediction for a VQA sample."""
    question:       str
    predicted_answer: str
    confidence:     float
    correct:        bool = False

    def to_dict(self) -> dict:
        return {"question": self.question, "predicted": self.predicted_answer,
                "confidence": round(self.confidence, 4), "correct": self.correct}


class EarlyFusionVQA(nn.Module):
    """
    LLaVA-style VQA: concatenate visual tokens + text tokens → LM.

    Args:
        vision_encoder: ViT encoder (returns cls_emb, patch_embs).
        lm:             Language model.
        cfg:            :class:`VQAConfig`.

    Example::

        model  = EarlyFusionVQA(vision_enc, lm, VQAConfig())
        imgs   = torch.randn(2, 3, 32, 32)
        q_ids  = torch.randint(0, 256, (2, 8))
        logits = model(imgs, q_ids)   # (2, n_visual+8, vocab_size)
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        lm:             nn.Module,
        cfg:            VQAConfig | None = None,
    ) -> None:
        super().__init__()
        self.vision_enc = vision_encoder
        self.lm         = lm
        self.cfg        = cfg or VQAConfig()
        self.projector  = VisualProjector(
            cfg.d_visual, cfg.d_model, n_layers=2
        )

    def forward(
        self,
        images:    torch.Tensor,
        q_ids:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: ``(B, C, H, W)`` images.
            q_ids:  ``(B, T)`` question token IDs.

        Returns:
            ``(B, N_vis + T, vocab_size)`` logits.
        """
        # 1. Encode image patches
        _, patch_embs = self.vision_enc(images)    # (B, N, d_visual)
        visual_tokens = self.projector(patch_embs) # (B, N, d_model)

        # 2. Encode text tokens
        # Get LM embeddings (first layer)
        if hasattr(self.lm, 'emb'):
            txt_tokens = self.lm.emb(q_ids % self.cfg.vocab_size)
        else:
            txt_tokens = q_ids.float().unsqueeze(-1).expand(
                *q_ids.shape, self.cfg.d_model
            )

        # 3. Concatenate: [visual | text]
        combined = torch.cat([visual_tokens, txt_tokens], dim=1)  # (B, N+T, d_model)

        # 4. Pass through LM (using GRU as LM backbone)
        if hasattr(self.lm, 'rnn'):
            h, _ = self.lm.rnn(combined)
            logits = self.lm.head(h)       # (B, N+T, vocab_size)
        else:
            out    = self.lm(combined.long().mean(dim=-1, keepdim=True).squeeze())
            logits = out[0] if isinstance(out, tuple) else out
        return logits

    @torch.no_grad()
    def generate_answer(
        self,
        images: torch.Tensor,
        q_ids:  torch.Tensor,
        max_len: int = 10,
    ) -> list[int]:
        """Greedy decoding to generate answer tokens."""
        logits = self.forward(images, q_ids)   # (B, N+T, V)
        # Take last position tokens as answer start
        generated = logits[:, -max_len:, :].argmax(dim=-1)
        return generated[0].tolist()


class CrossAttentionVQA(nn.Module):
    """
    Flamingo-style VQA: cross-attention between text and visual tokens.

    Args:
        vision_encoder: ViT encoder.
        lm:             Language model.
        cfg:            :class:`VQAConfig`.

    Example::

        model  = CrossAttentionVQA(vision_enc, lm, VQAConfig())
        imgs   = torch.randn(2, 3, 32, 32)
        q_ids  = torch.randint(0, 256, (2, 8))
        logits = model(imgs, q_ids)
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        lm:             nn.Module,
        cfg:            VQAConfig | None = None,
    ) -> None:
        super().__init__()
        self.vision_enc = vision_encoder
        self.lm         = lm
        self.cfg        = cfg or VQAConfig()
        xattn_cfg       = CrossAttentionConfig(
            d_model=cfg.d_model, d_visual=cfg.d_visual
        )
        self.cross_attn = CrossModalAttention(xattn_cfg)
        self.head       = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(
        self,
        images: torch.Tensor,
        q_ids:  torch.Tensor,
    ) -> torch.Tensor:
        _, patch_embs = self.vision_enc(images)  # (B, N, d_visual)

        # Encode text with LM
        if hasattr(self.lm, 'emb'):
            txt_emb = self.lm.emb(q_ids % self.cfg.vocab_size)
        else:
            txt_emb = torch.randn(*q_ids.shape, self.cfg.d_model)

        if hasattr(self.lm, 'rnn'):
            text_h, _ = self.lm.rnn(txt_emb)    # (B, T, d_model)
        else:
            text_h = txt_emb

        # Cross-attend text to visual
        fused = self.cross_attn(text_h, patch_embs)   # (B, T, d_model)
        return self.head(fused)                         # (B, T, vocab_size)


def vqa_accuracy(predictions: list[str], references: list[str]) -> float:
    """VQA accuracy: fraction of predictions that match any reference answer."""
    n_correct = sum(
        1 for pred, ref in zip(predictions, references)
        if pred.strip().lower() == ref.strip().lower()
    )
    return n_correct / max(len(predictions), 1)
