"""
nanomind/rag/retriever.py — Hybrid retriever and cross-encoder re-ranker.

## Two-Stage Retrieval

Stage 1 (Recall): Retrieve top-K candidates quickly
  - Dense: vector store cosine similarity
  - Sparse: BM25 keyword matching
  - Hybrid: combine both with RRF or linear interpolation

Stage 2 (Precision): Re-rank top-K with a cross-encoder
  - Cross-encoder sees full (query, passage) context
  - Much more accurate but slow (can't pre-compute)
  - Applied only to top-K from stage 1 (K=50-100)

## Reciprocal Rank Fusion (RRF)

Combine ranked lists without score normalisation:
  RRF_score(d) = Σ_{r∈rankings} 1 / (k + rank(d, r))

k=60 is standard (from Cormack et al., 2009).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from nanomind.rag.chunker import Chunk
from nanomind.rag.vector_store import VectorStore, SearchResult
from nanomind.rag.bm25 import BM25Retriever


@dataclass
class RetrievalResult:
    """Combined retrieval + re-ranking result."""
    chunk:       Chunk
    dense_score: float | None
    bm25_score:  float | None
    final_score: float
    rank:        int

    def to_dict(self) -> dict:
        return {
            "text":        self.chunk.text[:100],
            "doc_id":      self.chunk.doc_id,
            "final_score": round(self.final_score, 4),
            "rank":        self.rank,
        }


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]],
    k: int = 60,
) -> list[tuple[Chunk, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked chunk lists (each from a retriever).
        k:            RRF constant (default 60).

    Returns:
        List of (chunk, rrf_score) sorted by score descending.
    """
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            key = (chunk.doc_id, chunk.chunk_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    # Map key → chunk
    chunk_map: dict = {}
    for ranked in ranked_lists:
        for chunk in ranked:
            chunk_map[(chunk.doc_id, chunk.chunk_id)] = chunk

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    return [(chunk_map[k], s) for k, s in sorted_scores]


class HybridRetriever:
    """
    Two-stage hybrid retriever: dense + BM25 + RRF fusion.

    Args:
        vector_store:  :class:`VectorStore` for dense retrieval.
        bm25:          :class:`BM25Retriever` for sparse retrieval.
        alpha:         Weight for dense vs BM25 (0 = BM25 only, 1 = dense only).
        use_rrf:       Use RRF fusion instead of linear interpolation.

    Example::

        retriever = HybridRetriever(vector_store, bm25, alpha=0.5)
        results   = retriever.retrieve(query_embedding, "python list sorting", k=5)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25:         BM25Retriever,
        alpha:        float = 0.5,
        use_rrf:      bool  = True,
    ) -> None:
        self.vstore = vector_store
        self.bm25   = bm25
        self.alpha  = alpha
        self.use_rrf = use_rrf

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        query_text:      str,
        k:               int = 5,
        fetch_k:         int = 20,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-k chunks using hybrid search.

        Args:
            query_embedding: ``(D,)`` dense query vector.
            query_text:      Query text for BM25.
            k:               Final results to return.
            fetch_k:         Candidates to fetch from each retriever.

        Returns:
            List of :class:`RetrievalResult`.
        """
        # Stage 1: Dense retrieval
        dense_results = self.vstore.search(query_embedding, k=fetch_k)
        dense_chunks  = [r.chunk for r in dense_results]
        dense_map     = {(r.chunk.doc_id, r.chunk.chunk_id): r.score
                         for r in dense_results}

        # Stage 1: BM25 retrieval
        bm25_raw      = self.bm25.search(query_text, k=fetch_k)
        bm25_chunks   = [c for c, _ in bm25_raw]
        bm25_map      = {(c.doc_id, c.chunk_id): s for c, s in bm25_raw}

        if self.use_rrf:
            fused = reciprocal_rank_fusion([dense_chunks, bm25_chunks])[:k]
            results = []
            for rank, (chunk, score) in enumerate(fused):
                key = (chunk.doc_id, chunk.chunk_id)
                results.append(RetrievalResult(
                    chunk       = chunk,
                    dense_score = dense_map.get(key),
                    bm25_score  = bm25_map.get(key),
                    final_score = score,
                    rank        = rank,
                ))
        else:
            # Linear interpolation
            all_keys = set(dense_map) | set(bm25_map)
            scores   = []
            for key in all_keys:
                d = dense_map.get(key, 0.0)
                b = bm25_map.get(key, 0.0)
                scores.append((key, self.alpha * d + (1 - self.alpha) * b))
            scores.sort(key=lambda x: -x[1])
            chunk_map = {(c.doc_id, c.chunk_id): c
                         for c in dense_chunks + bm25_chunks}
            results = []
            for rank, (key, score) in enumerate(scores[:k]):
                if key in chunk_map:
                    results.append(RetrievalResult(
                        chunk       = chunk_map[key],
                        dense_score = dense_map.get(key),
                        bm25_score  = bm25_map.get(key),
                        final_score = score,
                        rank        = rank,
                    ))
        return results


class CrossEncoderReranker(nn.Module):
    """
    Cross-encoder re-ranker for stage-2 precision retrieval.

    Scores (query, passage) pairs jointly — much higher accuracy than bi-encoder.

    Args:
        backbone:  LM backbone.
        d_model:   Hidden dimension.

    Example::

        reranker = CrossEncoderReranker(backbone, d_model=128)
        scores   = reranker.score(query_ids, passage_ids_list)
        reranked = sorted(zip(passages, scores), key=lambda x: -x[1])
    """

    def __init__(self, backbone: nn.Module, d_model: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head     = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Score a batch of (query + passage) concatenations."""
        out    = self.backbone(input_ids)
        hidden = out[0] if isinstance(out, tuple) else out
        pooled = hidden[:, 0, :]   # CLS token
        return self.head(pooled).squeeze(-1)   # (B,)

    @torch.no_grad()
    def rerank(
        self,
        results:    list[RetrievalResult],
        query_ids:  list[int],
        max_len:    int = 32,
    ) -> list[RetrievalResult]:
        """Re-rank results using cross-encoder scores."""
        scores = []
        for r in results:
            # Concatenate query + passage tokens (simplified)
            passage_ids = [ord(c) % 256 for c in r.chunk.text[:max_len]]
            combined    = torch.tensor([query_ids + passage_ids]).long()
            score       = self.forward(combined).item()
            scores.append(score)

        ranked = sorted(zip(results, scores), key=lambda x: -x[1])
        reranked = []
        for rank, (res, score) in enumerate(ranked):
            reranked.append(RetrievalResult(
                chunk       = res.chunk,
                dense_score = res.dense_score,
                bm25_score  = res.bm25_score,
                final_score = score,
                rank        = rank,
            ))
        return reranked
