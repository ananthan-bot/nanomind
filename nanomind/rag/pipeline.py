"""
nanomind/rag/pipeline.py — Unified RAG pipeline: index → retrieve → augment.
"""

from __future__ import annotations

from nanomind.rag.config import RAGConfig
from nanomind.rag.types import Document, Chunk, RetrievalResult
from nanomind.rag.chunker import FixedSizeChunker
from nanomind.rag.embedder import TFIDFEmbedder, Embedder
from nanomind.rag.vector_store import VectorStore
from nanomind.rag.context import build_context, build_rag_prompt
from nanomind.utils.logger import get_logger

log = get_logger("rag.pipeline")


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Handles the full RAG workflow:
      1. ``index()``    — chunk + embed + store documents
      2. ``retrieve()`` — embed query + search vector store
      3. ``augment()``  — build context-augmented prompt

    Args:
        cfg:      RAG configuration.
        embedder: Text embedder (default: TF-IDF).
        chunker:  Document chunker (default: FixedSize).

    Example::

        pipeline = RAGPipeline()
        pipeline.index([Document("NanoMind is a LLM library...")])
        results = pipeline.retrieve("What is NanoMind?")
        prompt  = pipeline.augment("What is NanoMind?", results)
        # → feed prompt to LLM
    """

    def __init__(
        self,
        cfg:      RAGConfig  | None = None,
        embedder: Embedder   | None = None,
        chunker                     = None,
    ) -> None:
        self.cfg      = cfg or RAGConfig()
        self.chunker  = chunker or FixedSizeChunker(
            self.cfg.chunk_size, self.cfg.chunk_overlap
        )
        self.embedder = embedder or TFIDFEmbedder(
            max_features=self.cfg.embed_dim
        )
        self.store    = VectorStore(similarity=self.cfg.similarity)
        self._indexed = False

    def index(self, documents: list[Document]) -> "RAGPipeline":
        """
        Index a list of documents: chunk → embed → store.

        Args:
            documents: Source documents to index.

        Returns:
            Self (for chaining).
        """
        log.info(f"Indexing {len(documents)} documents...")
        chunks = self.chunker.chunk_many(documents)
        log.info(f"  → {len(chunks)} chunks created")

        # Fit embedder on all chunk texts, then embed
        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        self.embedder.embed_chunks(chunks)
        self.store.add(chunks)
        self._indexed = True
        log.info(f"  → {len(self.store)} chunks indexed")
        return self

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: User query string.
            top_k: Override default top-K from config.

        Returns:
            List of :class:`RetrievalResult`, sorted by score.
        """
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve().")
        k              = top_k or self.cfg.top_k
        query_emb      = self.embedder.embed(query)
        results        = self.store.search(
            query_emb, top_k=k, deduplicate=self.cfg.deduplicate
        )
        log.info(f"Retrieved {len(results)} chunks for query: {query[:60]!r}")
        return results

    def augment(
        self,
        query:   str,
        results: list[RetrievalResult] | None = None,
    ) -> str:
        """
        Build a RAG-augmented prompt for the query.

        Args:
            query:   User query string.
            results: Pre-retrieved results (auto-retrieves if None).

        Returns:
            Augmented prompt string with retrieved context.
        """
        if results is None:
            results = self.retrieve(query)
        return build_rag_prompt(
            query, results,
            template=self.cfg.context_template,
            max_context=self.cfg.max_context_len,
        )

    def query(self, query: str) -> tuple[str, list[RetrievalResult]]:
        """
        Full RAG query: retrieve + augment in one call.

        Returns:
            ``(augmented_prompt, retrieved_results)``
        """
        results = self.retrieve(query)
        prompt  = self.augment(query, results)
        return prompt, results

    def stats(self) -> dict:
        """Return pipeline statistics."""
        return {
            "n_documents": len(set(c.doc_id for c in self.store._chunks)),
            "n_chunks":    len(self.store),
            "embed_dim":   getattr(self.embedder, "embed_dim", self.cfg.embed_dim),
            "similarity":  self.cfg.similarity,
            "top_k":       self.cfg.top_k,
        }
