"""
nanomind/rag/vector_store.py — In-memory vector store for RAG retrieval.

Stores chunk embeddings and supports fast similarity search.

## Similarity Metrics

Cosine similarity:  cos(u, v) = u·v / (|u| |v|)
  - Range: [-1, 1], 1 = identical direction
  - Scale-invariant: only direction matters, not magnitude
  - Standard for text embeddings (sentence-transformers, OpenAI)

Dot product:        u·v = Σ u_i * v_i
  - Range: (-∞, +∞), larger = more similar
  - Fast (no normalisation), used by DPR, ColBERT

## Search Complexity

Brute-force: O(N·D) per query where N=corpus size, D=embedding dim
  Acceptable for N < 100k. For larger corpora use FAISS/HNSW.

FAISS (Facebook AI Similarity Search):
  - IVF index: O(sqrt(N)·D) — partitions vectors into clusters
  - HNSW:      O(log(N)·D) — hierarchical navigable small world graph
  NanoMind implements brute-force for educational clarity.
"""

from __future__ import annotations
import math
import json
from pathlib import Path

from nanomind.rag.types import Chunk, RetrievalResult


def _cosine(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorStore:
    """
    In-memory vector store: add chunks and search by embedding similarity.

    Args:
        similarity: Similarity function — ``"cosine"`` or ``"dot"``.

    Example::

        store = VectorStore()
        store.add(chunks)          # chunks with .embedding set
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(self, similarity: str = "cosine") -> None:
        self.similarity = similarity
        self._chunks:    list[Chunk]       = []
        self._embed_dim: int | None        = None
        self._sim_fn = _cosine if similarity == "cosine" else _dot

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks (with embeddings) to the store."""
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id!r} has no embedding. "
                                 "Call embedder.embed_chunks() first.")
            if self._embed_dim is None:
                self._embed_dim = len(chunk.embedding)
            self._chunks.append(chunk)

    def search(
        self,
        query_embedding: list[float],
        top_k:           int = 5,
        deduplicate:     bool = True,
    ) -> list[RetrievalResult]:
        """
        Find the top-K most similar chunks to the query embedding.

        Args:
            query_embedding: Dense query vector.
            top_k:           Number of results to return.
            deduplicate:     Skip duplicate chunk texts.

        Returns:
            List of :class:`RetrievalResult`, sorted by score descending.
        """
        if not self._chunks:
            return []
        scores = [(self._sim_fn(query_embedding, c.embedding), c)
                  for c in self._chunks if c.embedding]
        scores.sort(key=lambda x: -x[0])

        results, seen_texts = [], set()
        for score, chunk in scores:
            if deduplicate:
                key = chunk.text.strip()[:80]
                if key in seen_texts:
                    continue
                seen_texts.add(key)
            results.append(RetrievalResult(chunk=chunk, score=score, rank=len(results)+1))
            if len(results) >= top_k:
                break
        return results

    def __len__(self) -> int:
        return len(self._chunks)

    def save(self, path: str | Path) -> None:
        """Save store to a JSON file."""
        data = [{
            "text":       c.text,
            "doc_id":     c.doc_id,
            "chunk_id":   c.chunk_id,
            "start_char": c.start_char,
            "end_char":   c.end_char,
            "metadata":   c.metadata,
            "embedding":  c.embedding,
        } for c in self._chunks]
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: str | Path) -> "VectorStore":
        """Load store from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._chunks = [Chunk(**d) for d in data]
        if self._chunks and self._chunks[0].embedding:
            self._embed_dim = len(self._chunks[0].embedding)
        return self
