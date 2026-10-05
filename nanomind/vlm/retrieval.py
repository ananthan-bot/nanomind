"""
nanomind/vlm/retrieval.py — Cross-modal retrieval (image←→text search).

## Cross-Modal Retrieval Tasks

Image-to-Text Retrieval (I2T):
  Given an image, find the matching caption from a pool.
  "Retrieve the 5 most relevant captions for this image."

Text-to-Image Retrieval (T2I):
  Given a caption, find the matching image from a pool.
  "Find images showing a dog playing on a beach."

## Evaluation Metrics

Recall@K (R@K):
  "Is the correct match in the top K results?"
  R@1 = fraction of queries where correct is #1
  R@5 = fraction of queries where correct is in top 5
  R@10 = fraction of queries where correct is in top 10

Median Rank:
  Median rank of the correct item across all queries.
  Lower = better.

## CLIP for Retrieval

CLIP produces aligned image-text embeddings:
  1. Pre-encode all images/texts into embedding vectors
  2. At query time: embed query → cosine similarity search
  3. Return top-K matches

This enables O(1) per-query retrieval (after O(N) indexing).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    query_idx:  int
    result_idx: int
    score:      float
    rank:       int

    def to_dict(self) -> dict:
        return {"query": self.query_idx, "result": self.result_idx,
                "score": round(self.score, 4), "rank": self.rank}


@dataclass
class RetrievalMetrics:
    """Recall@K metrics for retrieval evaluation."""
    r_at_1:    float
    r_at_5:    float
    r_at_10:   float
    median_rank: float
    n_queries:  int

    def to_dict(self) -> dict:
        return {
            "R@1":         round(self.r_at_1, 4),
            "R@5":         round(self.r_at_5, 4),
            "R@10":        round(self.r_at_10, 4),
            "median_rank": self.median_rank,
            "n_queries":   self.n_queries,
        }


class CrossModalRetriever:
    """
    Cross-modal retrieval using CLIP-style embeddings.

    Supports:
      - Image-to-Text (I2T): query image → find matching text
      - Text-to-Image (T2I): query text → find matching image

    Args:
        clip_model: Trained :class:`CLIPModel`.

    Example::

        retriever = CrossModalRetriever(clip_model)
        retriever.index(images, text_ids)

        # Find top-5 captions for a query image
        results = retriever.image_to_text(query_image, k=5)
        # Find top-5 images for a query text
        results = retriever.text_to_image(query_text_ids, k=5)
    """

    def __init__(self, clip_model) -> None:
        self.clip = clip_model
        self._image_embs: torch.Tensor | None = None
        self._text_embs:  torch.Tensor | None = None
        self._n_items: int = 0

    @torch.no_grad()
    def index(
        self,
        images:    torch.Tensor,
        text_ids:  torch.Tensor,
        batch_size: int = 32,
    ) -> None:
        """
        Pre-encode and index all image-text pairs.

        Args:
            images:    ``(N, C, H, W)`` images.
            text_ids:  ``(N, T)`` text token IDs.
            batch_size: Batch size for encoding.
        """
        img_embs = []
        txt_embs = []
        N = images.shape[0]
        for i in range(0, N, batch_size):
            ib = images[i:i+batch_size]
            tb = text_ids[i:i+batch_size]
            img_embs.append(self.clip.encode_image(ib))
            txt_embs.append(self.clip.encode_text(tb))
        self._image_embs = torch.cat(img_embs, dim=0)
        self._text_embs  = torch.cat(txt_embs, dim=0)
        self._n_items    = N

    @torch.no_grad()
    def image_to_text(
        self,
        query_image: torch.Tensor,
        k:           int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve top-k texts for a query image."""
        if self._text_embs is None:
            raise RuntimeError("Call index() first")
        q    = self.clip.encode_image(query_image)          # (1, d)
        sims = (q @ self._text_embs.T).squeeze(0)          # (N,)
        return self._top_k(sims, k, query_idx=0)

    @torch.no_grad()
    def text_to_image(
        self,
        query_text: torch.Tensor,
        k:          int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve top-k images for a query text."""
        if self._image_embs is None:
            raise RuntimeError("Call index() first")
        q    = self.clip.encode_text(query_text)            # (1, d)
        sims = (q @ self._image_embs.T).squeeze(0)         # (N,)
        return self._top_k(sims, k, query_idx=0)

    def _top_k(
        self,
        scores:    torch.Tensor,
        k:         int,
        query_idx: int,
    ) -> list[RetrievalResult]:
        k = min(k, len(scores))
        top_scores, top_idx = scores.topk(k)
        return [
            RetrievalResult(
                query_idx  = query_idx,
                result_idx = idx.item(),
                score      = score.item(),
                rank       = rank,
            )
            for rank, (idx, score) in enumerate(
                zip(top_idx.tolist(), top_scores.tolist())
            )
        ]

    def evaluate(
        self,
        images:   torch.Tensor,
        text_ids: torch.Tensor,
        ks:       list[int] | None = None,
    ) -> RetrievalMetrics:
        """
        Evaluate I2T and T2I recall@K.

        Args:
            images:   ``(N, C, H, W)`` images.
            text_ids: ``(N, T)`` matching text IDs.
            ks:       List of K values for Recall@K.

        Returns:
            :class:`RetrievalMetrics`.
        """
        ks = ks or [1, 5, 10]
        self.index(images, text_ids)

        N    = self._n_items
        ranks_i2t = []
        ranks_t2i = []

        for i in range(N):
            # I2T: rank of correct text for image i
            q    = self._image_embs[i:i+1]
            sims = (q @ self._text_embs.T).squeeze(0)
            rank = (sims > sims[i]).sum().item()
            ranks_i2t.append(rank)

            # T2I: rank of correct image for text i
            q    = self._text_embs[i:i+1]
            sims = (q @ self._image_embs.T).squeeze(0)
            rank = (sims > sims[i]).sum().item()
            ranks_t2i.append(rank)

        all_ranks = ranks_i2t + ranks_t2i

        def recall_at(ranks, k):
            return sum(1 for r in ranks if r < k) / len(ranks)

        return RetrievalMetrics(
            r_at_1      = recall_at(all_ranks, 1),
            r_at_5      = recall_at(all_ranks, 5),
            r_at_10     = recall_at(all_ranks, 10),
            median_rank = float(sorted(all_ranks)[len(all_ranks) // 2] + 1),
            n_queries   = N,
        )
