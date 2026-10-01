"""
nanomind/rag/vector_store.py — In-memory vector store for RAG.

## Vector Stores

After embedding documents, store vectors for fast retrieval:
  - FAISS: Facebook's billion-scale similarity search
  - Chroma: Open-source vector DB with persistence
  - Pinecone: Cloud-hosted vector DB
  - Weaviate: Graph-aware vector DB
  - pgvector: PostgreSQL extension

NanoMind implements an in-memory flat vector store.
Production systems use FAISS with IVF (inverted file) indexing for O(√N) search.

## Similarity Search

Given query embedding q and corpus embeddings E:
  scores = E @ q   (dot product, if L2-normalised = cosine similarity)
  top_k  = argsort(scores, descending=True)[:k]

ANN (Approximate Nearest Neighbours):
  HNSW (Hierarchical Navigable Small World graphs): O(log N) search
  IVF (Inverted File Index): cluster vectors, search only relevant clusters
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from nanomind.rag.chunker import Chunk


@dataclass
class SearchResult:
    """A single retrieval result."""
    chunk:   Chunk
    score:   float
    rank:    int

    def to_dict(self) -> dict:
        return {"text": self.chunk.text, "score": round(self.score, 4),
                "doc_id": self.chunk.doc_id, "rank": self.rank}


class VectorStore:
    """
    In-memory flat vector store with cosine similarity search.

    Args:
        d_embed: Embedding dimension.

    Example::

        store = VectorStore(d_embed=64)
        store.add_chunks(chunks, embeddings)
        results = store.search(query_embedding, k=5)
    """

    def __init__(self, d_embed: int) -> None:
        self.d_embed  = d_embed
        self._chunks: list[Chunk]         = []
        self._embeds: list[torch.Tensor]  = []   # each (d_embed,)

    def add_chunks(
        self,
        chunks:     list[Chunk],
        embeddings: torch.Tensor,
    ) -> None:
        """
        Add chunks and their embeddings to the store.

        Args:
            chunks:     List of :class:`Chunk`.
            embeddings: ``(N, D)`` embedding tensor.
        """
        assert len(chunks) == embeddings.shape[0]
        for chunk, emb in zip(chunks, embeddings):
            self._chunks.append(chunk)
            self._embeds.append(F.normalize(emb, dim=-1))

    def search(
        self,
        query_embedding: torch.Tensor,
        k:               int = 5,
    ) -> list[SearchResult]:
        """
        Find k most similar chunks to query embedding.

        Args:
            query_embedding: ``(D,)`` or ``(1, D)`` query vector.
            k:               Number of results to return.

        Returns:
            List of :class:`SearchResult` sorted by score descending.
        """
        if not self._chunks:
            return []

        q      = F.normalize(query_embedding.flatten(), dim=-1)  # (D,)
        embeds = torch.stack(self._embeds, dim=0)                 # (N, D)
        scores = (embeds @ q).tolist()                            # (N,)

        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:k]
        return [
            SearchResult(chunk=self._chunks[i], score=s, rank=r)
            for r, (i, s) in enumerate(ranked)
        ]

    def search_batch(
        self,
        query_embeddings: torch.Tensor,
        k:                int = 5,
    ) -> list[list[SearchResult]]:
        """Search multiple queries in parallel."""
        return [self.search(q, k) for q in query_embeddings]

    def delete_doc(self, doc_id: str) -> int:
        """Remove all chunks from a document. Returns n removed."""
        before = len(self._chunks)
        pairs  = [(c, e) for c, e in zip(self._chunks, self._embeds)
                  if c.doc_id != doc_id]
        self._chunks = [p[0] for p in pairs]
        self._embeds = [p[1] for p in pairs]
        return before - len(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def stats(self) -> dict:
        docs = set(c.doc_id for c in self._chunks)
        return {
            "n_chunks": len(self._chunks),
            "n_docs":   len(docs),
            "d_embed":  self.d_embed,
        }
