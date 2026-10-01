"""
nanomind/rag/pipeline.py — End-to-end RAG pipeline.

Ties together: chunker → embedder → vector store → retriever → generator.

RAG Pipeline:
  Indexing (offline):
    1. Load documents
    2. Chunk into passages
    3. Embed each passage
    4. Store in vector store + BM25 index

  Retrieval + Generation (online):
    1. Embed query
    2. Retrieve top-K relevant passages
    3. Build prompt: context + question
    4. Generate answer with LLM

## Prompt Template for RAG

  "Answer the question based on the following context.
   If the answer is not in the context, say 'I don't know'.

   Context:
   [passage 1]
   [passage 2]
   ...

   Question: {question}
   Answer:"

This grounds the LLM answer in retrieved facts → reduces hallucination.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass, field

from nanomind.rag.chunker import Chunk, RecursiveChunker
from nanomind.rag.embedder import NanoEmbedder, EmbeddingResult
from nanomind.rag.vector_store import VectorStore, SearchResult
from nanomind.rag.bm25 import BM25Retriever
from nanomind.rag.retriever import HybridRetriever, RetrievalResult
from nanomind.utils.logger import get_logger

log = get_logger("rag.pipeline")


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""
    chunk_size:     int   = 512
    chunk_overlap:  int   = 64
    d_embed:        int   = 64
    top_k:          int   = 5
    fetch_k:        int   = 20
    alpha:          float = 0.5    # dense vs BM25 weight
    use_rrf:        bool  = True
    max_context_len: int  = 1024   # max chars from retrieved context


@dataclass
class RAGResult:
    """Result of a RAG query."""
    question:  str
    answer:    str
    retrieved: list[RetrievalResult]
    context:   str
    n_chunks_retrieved: int

    def to_dict(self) -> dict:
        return {
            "question":  self.question,
            "answer":    self.answer[:200],
            "n_chunks":  self.n_chunks_retrieved,
            "sources":   [r.chunk.doc_id for r in self.retrieved[:3]],
        }


class RAGPipeline:
    """
    End-to-end RAG pipeline: index documents, retrieve, generate.

    Args:
        embedder:   :class:`NanoEmbedder` for embedding.
        generator:  LM model for generation.
        cfg:        :class:`RAGConfig`.

    Example::

        pipeline = RAGPipeline(embedder, generator, RAGConfig())
        pipeline.index_document("Eiffel Tower is in Paris.", doc_id="wiki_1")
        result   = pipeline.query("Where is the Eiffel Tower?")
        print(result.answer)
    """

    PROMPT_TEMPLATE = (
        "Answer the question based only on the following context.
"
        "If the answer is not in the context, say 'I don't know'.

"
        "Context:
{context}

"
        "Question: {question}
"
        "Answer:"
    )

    def __init__(
        self,
        embedder:  NanoEmbedder,
        generator: nn.Module,
        cfg:       RAGConfig | None = None,
    ) -> None:
        self.embedder  = embedder
        self.generator = generator
        self.cfg       = cfg or RAGConfig()

        self.chunker  = RecursiveChunker(
            chunk_size = self.cfg.chunk_size,
            overlap    = self.cfg.chunk_overlap,
        )
        self.vstore   = VectorStore(d_embed=self.cfg.d_embed)
        self.bm25     = BM25Retriever()
        self._all_chunks: list[Chunk] = []
        self._bm25_indexed = False

    def index_document(
        self,
        text:     str,
        doc_id:   str = "doc",
        metadata: dict | None = None,
    ) -> int:
        """
        Chunk, embed, and index a document.

        Args:
            text:     Document text.
            doc_id:   Document identifier.
            metadata: Optional metadata dict.

        Returns:
            Number of chunks created.
        """
        chunks = self.chunker.chunk(text, doc_id=doc_id, metadata=metadata or {})
        if not chunks:
            return 0

        texts  = [c.text for c in chunks]
        result = self.embedder.encode(texts)
        self.vstore.add_chunks(chunks, result.embeddings)
        self._all_chunks.extend(chunks)
        self._bm25_indexed = False   # invalidate BM25 index
        log.info(f"Indexed {len(chunks)} chunks from '{doc_id}'")
        return len(chunks)

    def _ensure_bm25(self) -> BM25Retriever:
        if not self._bm25_indexed:
            self.bm25.index(self._all_chunks)
            self._bm25_indexed = True
        return self.bm25

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a question."""
        k = k or self.cfg.top_k
        q_emb     = self.embedder.encode([question]).embeddings[0]
        bm25      = self._ensure_bm25()
        retriever = HybridRetriever(
            self.vstore, bm25,
            alpha    = self.cfg.alpha,
            use_rrf  = self.cfg.use_rrf,
        )
        return retriever.retrieve(q_emb, question, k=k, fetch_k=self.cfg.fetch_k)

    def _build_context(self, results: list[RetrievalResult]) -> str:
        parts   = [r.chunk.text for r in results]
        context = "

".join(parts)
        return context[:self.cfg.max_context_len]

    def _generate(self, prompt: str) -> str:
        """Generate answer from prompt using the LM."""
        import torch.nn.functional as F
        tokens  = [ord(c) % 256 for c in prompt[:64]]
        ids     = torch.tensor([tokens]).long()
        with torch.no_grad():
            out    = self.generator(ids)
            logits = out[0] if isinstance(out, tuple) else out
            next_t = logits[0, -1, :].argmax().item()
        # Mock: return context-based answer for demo
        return f"[Generated answer based on {len(results_placeholder)} retrieved chunks]"

    def query(self, question: str) -> RAGResult:
        """
        Full RAG query: retrieve + generate.

        Args:
            question: User question.

        Returns:
            :class:`RAGResult`.
        """
        global results_placeholder
        retrieved  = self.retrieve(question)
        results_placeholder = retrieved
        context    = self._build_context(retrieved)
        prompt     = self.PROMPT_TEMPLATE.format(
            context=context, question=question
        )
        answer     = self._generate(prompt)
        return RAGResult(
            question            = question,
            answer              = answer,
            retrieved           = retrieved,
            context             = context,
            n_chunks_retrieved  = len(retrieved),
        )

    def stats(self) -> dict:
        return {
            "n_docs":   len(set(c.doc_id for c in self._all_chunks)),
            "n_chunks": len(self._all_chunks),
            **self.vstore.stats(),
        }

results_placeholder = []
