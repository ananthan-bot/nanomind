"""NanoMind RAG sub-package — Retrieval-Augmented Generation.

Implements the full RAG stack:
  1. Chunk / FixedSizeChunker     — fixed-size text chunking
  2. SentenceChunker              — sentence-boundary chunking
  3. RecursiveChunker             — LangChain-style recursive chunking
  4. NanoEmbedder                 — neural bi-encoder for dense retrieval
  5. TFIDFEmbedder                — sparse TF-IDF baseline
  6. EmbeddingResult              — embeddings + normalise()
  7. VectorStore                  — in-memory cosine similarity search
  8. SearchResult                 — chunk + score + rank
  9. BM25Retriever                — BM25 sparse retrieval
  10. HybridRetriever             — dense + BM25 + RRF fusion
  11. CrossEncoderReranker        — joint (query, passage) re-ranking
  12. RetrievalResult             — final retrieval result
  13. reciprocal_rank_fusion      — RRF score fusion
  14. RAGConfig                   — pipeline configuration
  15. RAGPipeline                 — index_document, retrieve, query
  16. RAGResult                   — question, answer, retrieved chunks

Primary exports:
    - :class:`Chunk`                 — chunked text with metadata
    - :class:`FixedSizeChunker`      — fixed-size chunking
    - :class:`SentenceChunker`       — sentence-boundary chunking
    - :class:`RecursiveChunker`      — recursive chunking
    - :class:`NanoEmbedder`          — neural text embedder
    - :class:`TFIDFEmbedder`         — TF-IDF baseline
    - :class:`EmbeddingResult`       — (N, D) embeddings
    - :class:`VectorStore`           — add_chunks, search, search_batch
    - :class:`SearchResult`          — chunk + cosine score
    - :class:`BM25Retriever`         — index, search (BM25)
    - :class:`HybridRetriever`       — dense+BM25, RRF or linear
    - :class:`CrossEncoderReranker`  — (query, passage) joint scoring
    - :class:`RetrievalResult`       — dense+bm25+final scores
    - :func:`reciprocal_rank_fusion` — RRF list merging
    - :class:`RAGConfig`             — pipeline config
    - :class:`RAGPipeline`           — index, retrieve, query
    - :class:`RAGResult`             — answer + retrieved chunks
"""

from nanomind.rag.chunker import Chunk, FixedSizeChunker, SentenceChunker, RecursiveChunker
from nanomind.rag.embedder import NanoEmbedder, TFIDFEmbedder, EmbeddingResult
from nanomind.rag.vector_store import VectorStore, SearchResult
from nanomind.rag.bm25 import BM25Retriever
from nanomind.rag.retriever import (
    HybridRetriever, CrossEncoderReranker, RetrievalResult,
    reciprocal_rank_fusion,
)
from nanomind.rag.pipeline import RAGConfig, RAGPipeline, RAGResult

__all__ = [
    "Chunk", "FixedSizeChunker", "SentenceChunker", "RecursiveChunker",
    "NanoEmbedder", "TFIDFEmbedder", "EmbeddingResult",
    "VectorStore", "SearchResult",
    "BM25Retriever",
    "HybridRetriever", "CrossEncoderReranker", "RetrievalResult",
    "reciprocal_rank_fusion",
    "RAGConfig", "RAGPipeline", "RAGResult",
]
