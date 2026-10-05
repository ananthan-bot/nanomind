"""
nanomind/vlm/captioning.py — Image captioning model.

## Image Captioning

Given an image, generate a natural language description.

Architecture:
  ViT encoder → visual features → LM decoder (autoregressive) → caption

Training: teacher forcing
  Input:  [BOS] + caption tokens
  Target: caption tokens + [EOS]
  Loss:   cross-entropy on target tokens only

## BLEU Score

Standard metric for captioning quality (also used for MT):
  BLEU-n = geometric mean of n-gram precisions × brevity penalty

  Modified precision:
    p_n = # n-gram matches / # n-grams in candidate (clipped to reference count)

  Geometric mean:
    BLEU = BP × exp(Σ wₙ × log pₙ)  where wₙ = 1/4, BP = brevity penalty

BLEU-4 is the standard metric for captioning benchmarks.

## CIDEr (Consensus-Based Image Description Evaluation)

Better metric for captioning than BLEU:
  - Weights n-grams by TF-IDF (rewards distinctive words)
  - Correlates better with human judgment

## Modern Captioning Models
  - BLIP-2: ViT + Q-Former + Flan-T5
  - LLaVA-1.5: ViT-L/336 + MLP + Vicuna-13B
  - Gemini: Proprietary multimodal
"""

from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.vlm.cross_attention import VisualProjector


@dataclass
class CaptioningConfig:
    """Configuration for image captioning model."""
    d_visual:   int   = 32
    d_model:    int   = 64
    vocab_size: int   = 256
    bos_token:  int   = 1
    eos_token:  int   = 2
    pad_token:  int   = 0
    max_len:    int   = 32


class ImageCaptioner(nn.Module):
    """
    Image captioning model: ViT encoder + autoregressive LM decoder.

    Args:
        vision_encoder: ViT image encoder.
        lm:             Language model (used as decoder).
        cfg:            :class:`CaptioningConfig`.

    Example::

        captioner = ImageCaptioner(vision_enc, lm, CaptioningConfig())
        imgs      = torch.randn(2, 3, 32, 32)
        tgt_ids   = torch.randint(0, 256, (2, 10))
        loss      = captioner.training_step(imgs, tgt_ids)
        captions  = captioner.generate(imgs[:1], max_new_tokens=10)
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        lm:             nn.Module,
        cfg:            CaptioningConfig | None = None,
    ) -> None:
        super().__init__()
        self.vision_enc = vision_encoder
        self.lm         = lm
        self.cfg        = cfg or CaptioningConfig()
        self.projector  = VisualProjector(cfg.d_visual, cfg.d_model)

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to visual token sequence."""
        _, patch_embs = self.vision_enc(images)
        return self.projector(patch_embs)   # (B, N, d_model)

    def training_step(
        self,
        images:  torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute teacher-forced captioning loss.

        Args:
            images:  ``(B, C, H, W)`` images.
            tgt_ids: ``(B, T)`` target caption token IDs.

        Returns:
            Scalar cross-entropy loss.
        """
        visual_tokens = self._encode_image(images)   # (B, N, d_model)

        # Get text embeddings
        if hasattr(self.lm, 'emb'):
            txt_emb = self.lm.emb(tgt_ids[:, :-1] % self.cfg.vocab_size)
        else:
            txt_emb = torch.zeros(*tgt_ids[:, :-1].shape, self.cfg.d_model)

        # Concat visual prefix + text input
        decoder_input = torch.cat([visual_tokens, txt_emb], dim=1)

        # Run through LM decoder
        if hasattr(self.lm, 'rnn'):
            h, _ = self.lm.rnn(decoder_input)
        else:
            h = decoder_input

        # Predict next tokens (only on text portion)
        N       = visual_tokens.shape[1]
        text_h  = h[:, N:, :]            # (B, T-1, d_model)

        if hasattr(self.lm, 'head'):
            logits = self.lm.head(text_h)  # (B, T-1, V)
        else:
            logits = text_h

        target = tgt_ids[:, 1:].clamp(0, self.cfg.vocab_size - 1)
        loss   = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            target.reshape(-1),
        )
        return loss

    @torch.no_grad()
    def generate(
        self,
        images:         torch.Tensor,
        max_new_tokens: int = 20,
    ) -> list[int]:
        """Greedy decoding to generate a caption."""
        visual_tokens = self._encode_image(images)   # (1, N, d_model)
        generated     = [self.cfg.bos_token]

        for _ in range(max_new_tokens):
            tgt = torch.tensor([generated]).long()
            if hasattr(self.lm, 'emb'):
                txt_emb = self.lm.emb(tgt % self.cfg.vocab_size)
            else:
                txt_emb = torch.zeros(1, len(generated), self.cfg.d_model)

            dec_in = torch.cat([visual_tokens, txt_emb], dim=1)
            if hasattr(self.lm, 'rnn'):
                h, _ = self.lm.rnn(dec_in)
            else:
                h = dec_in

            last_h = h[:, -1, :]  # (1, d_model)
            if hasattr(self.lm, 'head'):
                logits = self.lm.head(last_h)   # (1, V)
            else:
                logits = torch.randn(1, self.cfg.vocab_size)

            next_token = logits.argmax(dim=-1).item()
            generated.append(next_token)
            if next_token == self.cfg.eos_token:
                break

        return generated


def bleu_score(
    candidate: list[str],
    references: list[list[str]],
    max_n:     int = 4,
) -> float:
    """
    Compute BLEU-n score.

    Args:
        candidate:  Tokenised candidate (list of words).
        references: List of tokenised references.
        max_n:      Maximum n-gram order.

    Returns:
        BLEU score in [0, 1].
    """
    if not candidate:
        return 0.0

    # Brevity penalty
    ref_len  = min(len(r) for r in references)
    cand_len = len(candidate)
    bp       = 1.0 if cand_len >= ref_len else math.exp(1 - ref_len / cand_len)

    # Modified n-gram precision
    log_sum = 0.0
    for n in range(1, max_n + 1):
        cand_ngrams = Counter(tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1))
        if not cand_ngrams:
            return 0.0
        clipped = 0
        for ngram, count in cand_ngrams.items():
            max_ref = max(
                Counter(tuple(r[i:i+n]) for i in range(len(r)-n+1)).get(ngram, 0)
                for r in references
            )
            clipped += min(count, max_ref)
        total = sum(cand_ngrams.values())
        if total == 0 or clipped == 0:
            return 0.0
        log_sum += math.log(clipped / total) / max_n

    return bp * math.exp(log_sum)
