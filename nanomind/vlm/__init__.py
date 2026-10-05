"""NanoMind VLM sub-package — Vision-Language Models.

Implements the full VLM stack:
  1. VisionEncoderConfig       — image size, patch size, d_model, n_layers
  2. PatchEmbedding            — patchify + project + CLS token + pos embed
  3. VisionTransformerBlock    — Pre-LN self-attention + FFN
  4. VisionEncoder             — full ViT: embed → transformer → project
  5. CLIPConfig                — temperature, d_embed
  6. CLIPLoss                  — symmetric InfoNCE contrastive loss
  7. CLIPModel                 — encode_image/text, zero_shot_classify
  8. CrossAttentionConfig      — d_model, d_visual, n_heads
  9. PerceiverResampler        — compress N patches to fixed visual tokens
  10. VisualProjector          — MLP projector (ViT dim → LM dim)
  11. CrossModalAttention      — text attends to visual tokens (Flamingo)
  12. VQAConfig / VQASample    — VQA data structures
  13. EarlyFusionVQA           — LLaVA-style (visual + text tokens concatenated)
  14. CrossAttentionVQA        — Flamingo-style (cross-attention layers)
  15. vqa_accuracy             — VQA evaluation metric
  16. CaptioningConfig         — captioning model config
  17. ImageCaptioner           — ViT + LM decoder, training_step, generate
  18. bleu_score               — BLEU-4 captioning metric
  19. CrossModalRetriever      — image/text search with CLIP embeddings
  20. RetrievalMetrics         — Recall@1/5/10, median rank

Primary exports:
    - :class:`VisionEncoderConfig`   — ViT configuration
    - :class:`PatchEmbedding`        — image → patch tokens
    - :class:`VisionEncoder`         — full ViT encoder
    - :class:`CLIPLoss`              — symmetric contrastive loss
    - :class:`CLIPModel`             — full CLIP (encode + loss)
    - :class:`PerceiverResampler`    — Flamingo-style token compression
    - :class:`VisualProjector`       — LLaVA-style MLP projector
    - :class:`CrossModalAttention`   — text-visual cross-attention
    - :class:`EarlyFusionVQA`        — LLaVA-style VQA model
    - :class:`CrossAttentionVQA`     — Flamingo-style VQA model
    - :func:`vqa_accuracy`           — VQA eval metric
    - :class:`ImageCaptioner`        — image → caption model
    - :func:`bleu_score`             — BLEU-4 metric
    - :class:`CrossModalRetriever`   — image/text retrieval + Recall@K
    - :class:`RetrievalMetrics`      — R@1, R@5, R@10, median rank
"""

from nanomind.vlm.vision_encoder import (
    VisionEncoderConfig, PatchEmbedding, VisionTransformerBlock, VisionEncoder,
)
from nanomind.vlm.clip import CLIPConfig, CLIPLoss, CLIPModel
from nanomind.vlm.cross_attention import (
    CrossAttentionConfig, PerceiverResampler, VisualProjector, CrossModalAttention,
)
from nanomind.vlm.vqa import (
    VQAConfig, VQASample, VQAResult, EarlyFusionVQA, CrossAttentionVQA, vqa_accuracy,
)
from nanomind.vlm.captioning import CaptioningConfig, ImageCaptioner, bleu_score
from nanomind.vlm.retrieval import (
    CrossModalRetriever, RetrievalMetrics, RetrievalResult,
)

__all__ = [
    "VisionEncoderConfig", "PatchEmbedding", "VisionTransformerBlock", "VisionEncoder",
    "CLIPConfig", "CLIPLoss", "CLIPModel",
    "CrossAttentionConfig", "PerceiverResampler", "VisualProjector", "CrossModalAttention",
    "VQAConfig", "VQASample", "VQAResult", "EarlyFusionVQA", "CrossAttentionVQA",
    "vqa_accuracy",
    "CaptioningConfig", "ImageCaptioner", "bleu_score",
    "CrossModalRetriever", "RetrievalMetrics", "RetrievalResult",
]
