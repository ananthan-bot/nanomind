"""NanoMind RAG sub-package — Retrieval-Augmented Generation.

Full RAG pipeline in three steps:
  1. index()    — chunk + embed + store documents
  2. retrieve() — embed query + cosine/dot search
  3. augment()  — inject context into prompt for the LLM

Primary exports:
    - :class:`RAGPipeline`        — end-to-end index/retrieve/augment/query
    - :class:`RAGConfig`          — chunk_size, top_k, embed_dim, similarity
    - :class:`Document`           — source document with text + metadata
    - :class:`Chunk`              — text sub-section, unit of retrieval
    - :class:`RetrievalResult`    — (chunk, score, rank) search result
    - :class:`FixedSizeChunker`   — fixed-window character chunking
    - :class:`SentenceChunker`    — sentence-boundary chunking
    - :class:`ParagraphChunker`   — blank-line paragraph chunking
    - :class:`TFIDFEmbedder`      — TF-IDF sparse embedder
    - :class:`BM25Embedder`       — BM25 sparse embedder
    - :class:`DenseEmbedder`      — NanoMind neural dense embedder
    - :class:`VectorStore`        — cosine/dot similarity search + save/load
    - :func:`build_context`       — format retrieved chunks into context
    - :func:`build_rag_prompt`    — build full augmented prompt
    - :func:`load_text_file`      — load .txt document
    - :func:`load_markdown_file`  — load .md document (strips syntax)
    - :func:`load_string`         — create Document from string
    - :func:`load_strings`        — create Documents from list of strings
"""

from nanomind.rag.config import RAGConfig
from nanomind.rag.types import Document, Chunk, RetrievalResult
from nanomind.rag.chunker import FixedSizeChunker, SentenceChunker, ParagraphChunker
from nanomind.rag.embedder import TFIDFEmbedder, BM25Embedder, DenseEmbedder
from nanomind.rag.vector_store import VectorStore
from nanomind.rag.context import build_context, build_rag_prompt
from nanomind.rag.pipeline import RAGPipeline
from nanomind.rag.loaders import (
    load_text_file, load_markdown_file, load_string, load_strings
)

__all__ = [
    "RAGConfig", "RAGPipeline",
    "Document", "Chunk", "RetrievalResult",
    "FixedSizeChunker", "SentenceChunker", "ParagraphChunker",
    "TFIDFEmbedder", "BM25Embedder", "DenseEmbedder",
    "VectorStore",
    "build_context", "build_rag_prompt",
    "load_text_file", "load_markdown_file", "load_string", "load_strings",
]
