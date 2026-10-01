"""tests/test_rag.py — Tests for NanoMind RAG package."""
import pytest
import torch
import torch.nn as nn
from nanomind.rag import (
    Chunk, FixedSizeChunker, SentenceChunker, RecursiveChunker,
    NanoEmbedder, TFIDFEmbedder, EmbeddingResult,
    VectorStore, SearchResult,
    BM25Retriever,
    HybridRetriever, CrossEncoderReranker,
    reciprocal_rank_fusion,
    RAGConfig, RAGPipeline, RAGResult,
)

V = 64

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 8)
        self.rnn  = nn.GRU(8, 16, batch_first=True)
        self.head = nn.Linear(16, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x % V))
        return self.head(h), None

DOCS = [
    "The Eiffel Tower is in Paris France.",
    "Python is a programming language created by Guido.",
    "The Amazon River is the largest river by volume.",
    "Machine learning uses statistical techniques.",
]


# ── Chunkers ──────────────────────────────────────────────────────────────────

class TestFixedSizeChunker:
    def test_basic_chunking(self):
        c = FixedSizeChunker(chunk_size=20, overlap=5)
        chunks = c.chunk("A" * 50, "doc1")
        assert len(chunks) > 1

    def test_single_chunk(self):
        c = FixedSizeChunker(chunk_size=100)
        chunks = c.chunk("Short text.", "doc1")
        assert len(chunks) == 1

    def test_chunk_has_metadata(self):
        c = FixedSizeChunker(chunk_size=20)
        chunks = c.chunk("x" * 40, "doc1", metadata={"author": "test"})
        assert chunks[0].metadata["author"] == "test"


class TestSentenceChunker:
    def test_splits_sentences(self):
        c = SentenceChunker(max_sentences=2)
        chunks = c.chunk("First sentence. Second one. Third one.", "d")
        assert len(chunks) >= 1

    def test_doc_id_preserved(self):
        c = SentenceChunker()
        chunks = c.chunk("Hello world.", "my_doc")
        assert all(ch.doc_id == "my_doc" for ch in chunks)


class TestRecursiveChunker:
    def test_produces_chunks(self):
        c = RecursiveChunker(chunk_size=30, overlap=5)
        chunks = c.chunk("Para one.

Para two.

Para three.", "d")
        assert len(chunks) >= 1

    def test_chunk_size_respected(self):
        c = RecursiveChunker(chunk_size=20, overlap=0)
        chunks = c.chunk("word " * 20, "d")
        for ch in chunks:
            assert ch.n_chars <= 40   # some leeway


# ── Embedders ─────────────────────────────────────────────────────────────────

class TestNanoEmbedder:
    def _emb(self):
        return NanoEmbedder(TinyLM(), d_model=V, d_embed=16, normalise=True)

    def test_encode_shape(self):
        e = self._emb()
        r = e.encode(["hello", "world"])
        assert r.embeddings.shape == (2, 16)

    def test_normalised(self):
        e = self._emb()
        r = e.encode(["test sentence"])
        norms = r.embeddings.norm(dim=-1)
        assert all(abs(n.item() - 1.0) < 0.01 for n in norms)

    def test_len(self):
        e = self._emb()
        r = e.encode(["a", "b", "c"])
        assert len(r) == 3


class TestTFIDFEmbedder:
    def test_encode_shape(self):
        e = TFIDFEmbedder(vocab_size=50)
        e.fit(DOCS)
        r = e.encode(DOCS[:2])
        assert r.embeddings.shape[0] == 2

    def test_fit_and_encode(self):
        e = TFIDFEmbedder(vocab_size=100)
        e.fit(DOCS)
        r = e.encode(["Eiffel Tower Paris"])
        assert r.embeddings.shape[0] == 1


# ── VectorStore ───────────────────────────────────────────────────────────────

class TestVectorStore:
    def _store_with_data(self):
        e = NanoEmbedder(TinyLM(), V, 16)
        chunks = [Chunk(t, f"d{i}", 0, 0, len(t)) for i, t in enumerate(DOCS)]
        embs   = e.encode(DOCS).embeddings
        s      = VectorStore(d_embed=16)
        s.add_chunks(chunks, embs)
        return s, e

    def test_len(self):
        s, _ = self._store_with_data()
        assert len(s) == len(DOCS)

    def test_search_returns_results(self):
        s, e = self._store_with_data()
        q    = e.encode(["Paris"]).embeddings[0]
        res  = s.search(q, k=2)
        assert len(res) == 2

    def test_search_scores_sorted(self):
        s, e = self._store_with_data()
        q    = e.encode(["river"]).embeddings[0]
        res  = s.search(q, k=3)
        scores = [r.score for r in res]
        assert scores == sorted(scores, reverse=True)

    def test_delete_doc(self):
        s, _ = self._store_with_data()
        removed = s.delete_doc("d0")
        assert removed == 1
        assert len(s) == len(DOCS) - 1

    def test_stats_keys(self):
        s, _ = self._store_with_data()
        st   = s.stats()
        assert "n_chunks" in st and "n_docs" in st


# ── BM25 ──────────────────────────────────────────────────────────────────────

class TestBM25:
    def _bm25(self):
        chunks = [Chunk(t, f"d{i}", 0, 0, len(t)) for i, t in enumerate(DOCS)]
        b = BM25Retriever()
        b.index(chunks)
        return b

    def test_search_returns_pairs(self):
        b   = self._bm25()
        res = b.search("Paris Eiffel Tower", k=2)
        assert len(res) == 2
        assert all(isinstance(c, Chunk) for c, _ in res)

    def test_relevant_doc_scores_high(self):
        b   = self._bm25()
        res = b.search("Paris Eiffel Tower", k=4)
        top_text = res[0][0].text.lower()
        assert "paris" in top_text or "eiffel" in top_text

    def test_not_indexed_raises(self):
        b = BM25Retriever()
        with pytest.raises(RuntimeError):
            b.search("test")


# ── Hybrid & RRF ──────────────────────────────────────────────────────────────

class TestRRF:
    def test_rrf_merges(self):
        chunks = [Chunk(t, f"d{i}", i, 0, len(t)) for i, t in enumerate(DOCS)]
        fused  = reciprocal_rank_fusion([chunks[:3], chunks[1:]], k=60)
        assert len(fused) > 0

    def test_rrf_score_positive(self):
        chunks = [Chunk("a", "d0", 0, 0, 1), Chunk("b", "d1", 0, 0, 1)]
        fused  = reciprocal_rank_fusion([chunks, chunks[::-1]])
        assert all(s > 0 for _, s in fused)


# ── RAGPipeline ───────────────────────────────────────────────────────────────

class TestRAGPipeline:
    def _pipeline(self):
        e   = NanoEmbedder(TinyLM(), V, 16)
        cfg = RAGConfig(chunk_size=50, chunk_overlap=10, d_embed=16, top_k=2)
        return RAGPipeline(e, TinyLM(), cfg)

    def test_index_document(self):
        p = self._pipeline()
        n = p.index_document("Hello world this is a test sentence.", "d1")
        assert n >= 1

    def test_retrieve_returns_results(self):
        p = self._pipeline()
        for i, t in enumerate(DOCS):
            p.index_document(t, f"d{i}")
        res = p.retrieve("Eiffel Tower Paris", k=2)
        assert len(res) <= 2

    def test_stats_after_indexing(self):
        p = self._pipeline()
        p.index_document("Some text", "d1")
        s = p.stats()
        assert s["n_docs"] == 1

    def test_query_returns_rag_result(self):
        p = self._pipeline()
        for i, t in enumerate(DOCS):
            p.index_document(t, f"d{i}")
        r = p.query("What is the largest river?")
        assert isinstance(r, RAGResult)
        assert r.question == "What is the largest river?"
