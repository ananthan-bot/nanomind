"""
nanomind/rag/config.py — RAG pipeline configuration.

## What is RAG?

Retrieval-Augmented Generation (RAG) solves a key limitation of language models:

  Problem:  LLMs have a fixed knowledge cutoff and cannot access private documents.
  Solution: At inference time, retrieve relevant documents from a corpus and inject
            them into the prompt before generating a response.

RAG Pipeline:
  1. Indexing (offline):
       Document → Chunker → Chunks → Embedder → Vectors → VectorStore

  2. Retrieval (online):
       Query → Embedder → Query Vector → VectorStore.search() → Top-K Chunks

  3. Augmented Generation:
       Query + Retrieved Chunks → PromptTemplate → LLM → Response

Key papers:
  Lewis et al. (2020) "Retrieval-Augmented Generation for Knowledge-Intensive NLP"
  https://arxiv.org/abs/2005.11401

  Gao et al. (2023) "Precise Zero-Shot Dense Retrieval without Relevance Labels"
  https://arxiv.org/abs/2212.10496
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    """
    Configuration for the RAG pipeline.

    Attributes:
        chunk_size:      Maximum characters per text chunk.
        chunk_overlap:   Characters of overlap between consecutive chunks.
        top_k:           Number of chunks to retrieve per query.
        embed_dim:       Embedding dimension for the vector store.
        similarity:      Similarity metric: ``"cosine"`` or ``"dot"``.
        context_template: Template for inserting retrieved context into prompts.
        max_context_len: Maximum characters of retrieved context to include.
        deduplicate:     Remove duplicate chunks from retrieval results.
    """
    chunk_size:       int   = 512
    chunk_overlap:    int   = 64
    top_k:            int   = 5
    embed_dim:        int   = 128
    similarity:       str   = "cosine"
    context_template: str   = "Context:
{context}

Question: {query}"
    max_context_len:  int   = 2048
    deduplicate:      bool  = True

    def __post_init__(self) -> None:
        assert self.chunk_size    >  0
        assert self.chunk_overlap >= 0
        assert self.chunk_overlap <  self.chunk_size
        assert self.top_k         >= 1
        assert self.embed_dim     >= 1
        assert self.similarity in ("cosine", "dot")
