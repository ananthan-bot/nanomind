"""tests/test_rag.py — Tests for the NanoMind RAG pipeline."""
import json, pytest, tempfile
from pathlib import Path
from nanomind.rag import (
    RAGConfig, RAGPipeline,
    Document, Chunk, RetrievalResult,
    FixedSizeChunker, SentenceChunker, ParagraphChunker,
    TFIDFEmbedder, BM25Embedder,
    VectorStore,
    build_context, build_rag_prompt,
    load_string, load_strings,
)

# Helpers
def doc(text="hello world foo bar baz", title="test"):
    return Document(text=text, title=title)

CORPUS = [
    doc("NanoMind is a language model library built from scratch in 30 days."),
    doc("Transformers use self-attention to process sequences in parallel."),
    doc("RAG retrieves relevant documents before generating an answer."),
    doc("LoRA fine-tunes models with a fraction of the trainable parameters."),
]


# ── RAGConfig ─────────────────────────────────────────────────────────────────

class TestRAGConfig:
    def test_defaults(self):
        cfg = RAGConfig()
        assert cfg.chunk_size == 512
        assert cfg.top_k == 5

    def test_invalid_overlap(self):
        with pytest.raises(AssertionError):
            RAGConfig(chunk_size=100, chunk_overlap=200)

    def test_invalid_similarity(self):
        with pytest.raises(AssertionError):
            RAGConfig(similarity="l2")


# ── Document / Chunk ──────────────────────────────────────────────────────────

class TestTypes:
    def test_document_len(self):
        d = doc("hello world")
        assert len(d) == 11

    def test_chunk_len(self):
        c = Chunk(text="hi", doc_id="x")
        assert len(c) == 2

    def test_retrieval_result_repr(self):
        c = Chunk(text="test", doc_id="d")
        r = RetrievalResult(chunk=c, score=0.9, rank=1)
        assert "0.9" in repr(r)


# ── Chunkers ──────────────────────────────────────────────────────────────────

class TestChunkers:
    def test_fixed_creates_chunks(self):
        c      = FixedSizeChunker(chunk_size=20, chunk_overlap=5)
        chunks = c.chunk(doc("a " * 30))
        assert len(chunks) > 1

    def test_fixed_chunk_text_length(self):
        c      = FixedSizeChunker(chunk_size=20, chunk_overlap=0)
        chunks = c.chunk(doc("x" * 60))
        for ch in chunks[:-1]:
            assert len(ch.text) <= 20

    def test_sentence_chunker(self):
        text = ("First sentence. Second sentence! Third sentence? "
                "Fourth sentence. Fifth sentence.")
        c    = SentenceChunker(chunk_size=40, chunk_overlap=0)
        chunks = c.chunk(doc(text))
        assert len(chunks) >= 1

    def test_paragraph_chunker(self):
        text = "Para one content.

Para two content.

Para three."
        c    = ParagraphChunker(chunk_size=30)
        chunks = c.chunk(doc(text))
        assert len(chunks) >= 1

    def test_chunk_many(self):
        c      = FixedSizeChunker(chunk_size=50, chunk_overlap=0)
        chunks = c.chunk_many(CORPUS)
        assert len(chunks) >= len(CORPUS)


# ── Embedders ─────────────────────────────────────────────────────────────────

class TestEmbedders:
    def _fitted_tfidf(self):
        e = TFIDFEmbedder(max_features=32)
        e.fit([d.text for d in CORPUS])
        return e

    def test_tfidf_fit_sets_vocab(self):
        e = self._fitted_tfidf()
        assert len(e._vocab) > 0

    def test_tfidf_embed_dim(self):
        e   = self._fitted_tfidf()
        vec = e.embed("language model")
        assert len(vec) == e.embed_dim

    def test_tfidf_embed_normalised(self):
        import math
        e   = self._fitted_tfidf()
        vec = e.embed("NanoMind library")
        norm = math.sqrt(sum(v*v for v in vec))
        assert abs(norm - 1.0) < 1e-5 or norm < 1e-9  # zero or unit norm

    def test_bm25_embed_returns_vector(self):
        e = BM25Embedder(max_features=32)
        e.fit([d.text for d in CORPUS])
        vec = e.embed("transformer attention")
        assert len(vec) == e.embed_dim

    def test_embed_many(self):
        e    = self._fitted_tfidf()
        vecs = e.embed_many(["hello", "world"])
        assert len(vecs) == 2

    def test_embed_chunks(self):
        e      = self._fitted_tfidf()
        chunks = [Chunk(text=d.text, doc_id=d.doc_id) for d in CORPUS]
        e.embed_chunks(chunks)
        for c in chunks:
            assert c.embedding is not None


# ── VectorStore ───────────────────────────────────────────────────────────────

class TestVectorStore:
    def _store_with_chunks(self):
        e      = TFIDFEmbedder(max_features=32)
        texts  = [d.text for d in CORPUS]
        e.fit(texts)
        chunks = [Chunk(text=d.text, doc_id=d.doc_id) for d in CORPUS]
        e.embed_chunks(chunks)
        store  = VectorStore()
        store.add(chunks)
        return store, e

    def test_len_after_add(self):
        store, _ = self._store_with_chunks()
        assert len(store) == len(CORPUS)

    def test_search_returns_results(self):
        store, e = self._store_with_chunks()
        q_emb    = e.embed("language model library")
        results  = store.search(q_emb, top_k=2)
        assert len(results) == 2

    def test_search_rank_order(self):
        store, e = self._store_with_chunks()
        q_emb    = e.embed("transformer attention")
        results  = store.search(q_emb, top_k=3)
        scores   = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_store(self):
        store  = VectorStore()
        result = store.search([0.0] * 10, top_k=3)
        assert result == []

    def test_save_load(self, tmp_path):
        store, _ = self._store_with_chunks()
        path = tmp_path / "store.json"
        store.save(path)
        store2 = VectorStore()
        store2.load(path)
        assert len(store2) == len(store)

    def test_no_embedding_raises(self):
        store = VectorStore()
        with pytest.raises(ValueError):
            store.add([Chunk(text="hi", doc_id="x")])


# ── Context builder ───────────────────────────────────────────────────────────

class TestContext:
    def _results(self):
        chunks = [
            Chunk(text="NanoMind is great.", doc_id="doc1"),
            Chunk(text="RAG retrieves context.", doc_id="doc2"),
        ]
        return [RetrievalResult(chunk=c, score=0.9-i*0.1, rank=i+1)
                for i, c in enumerate(chunks)]

    def test_build_context_contains_text(self):
        ctx = build_context(self._results())
        assert "NanoMind" in ctx

    def test_build_context_max_chars(self):
        ctx = build_context(self._results(), max_chars=20)
        assert len(ctx) <= 40  # some slack for header

    def test_build_rag_prompt_contains_query(self):
        prompt = build_rag_prompt("What is RAG?", self._results())
        assert "What is RAG?" in prompt

    def test_build_rag_prompt_contains_context(self):
        prompt = build_rag_prompt("question", self._results())
        assert "NanoMind" in prompt or "RAG" in prompt


# ── RAGPipeline ───────────────────────────────────────────────────────────────

class TestRAGPipeline:
    def _pipeline(self):
        cfg = RAGConfig(chunk_size=100, top_k=2, embed_dim=32)
        p   = RAGPipeline(cfg=cfg, embedder=TFIDFEmbedder(max_features=32))
        p.index(CORPUS)
        return p

    def test_index_creates_chunks(self):
        p = self._pipeline()
        assert len(p.store) > 0

    def test_retrieve_returns_results(self):
        p       = self._pipeline()
        results = p.retrieve("language model")
        assert len(results) > 0

    def test_retrieve_before_index_raises(self):
        p = RAGPipeline()
        with pytest.raises(RuntimeError):
            p.retrieve("query")

    def test_augment_contains_query(self):
        p      = self._pipeline()
        prompt = p.augment("What is NanoMind?")
        assert "NanoMind" in prompt or "What is NanoMind?" in prompt

    def test_query_returns_tuple(self):
        p             = self._pipeline()
        prompt, results = p.query("attention mechanism")
        assert isinstance(prompt, str)
        assert isinstance(results, list)

    def test_stats_keys(self):
        p     = self._pipeline()
        stats = p.stats()
        for k in ("n_documents", "n_chunks", "embed_dim", "similarity"):
            assert k in stats

    def test_top_k_respected(self):
        p       = self._pipeline()
        results = p.retrieve("transformer", top_k=1)
        assert len(results) == 1


# ── Loaders ───────────────────────────────────────────────────────────────────

class TestLoaders:
    def test_load_string(self):
        d = load_string("hello world", title="test")
        assert d.text == "hello world"
        assert d.title == "test"

    def test_load_strings(self):
        docs = load_strings(["a", "b", "c"])
        assert len(docs) == 3
        assert docs[0].text == "a"

    def test_load_strings_custom_titles(self):
        docs = load_strings(["x", "y"], titles=["X", "Y"])
        assert docs[0].title == "X"

    def test_load_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello from file", encoding="utf-8")
        from nanomind.rag import load_text_file
        d = load_text_file(f)
        assert "hello from file" in d.text

    def test_load_markdown_strips_headers(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title

Some content.", encoding="utf-8")
        from nanomind.rag import load_markdown_file
        d = load_markdown_file(f)
        assert "#" not in d.text
        assert "Title" in d.text


# ── Dot similarity ────────────────────────────────────────────────────────────

class TestDotSimilarity:
    def test_dot_store_returns_results(self):
        e      = TFIDFEmbedder(max_features=16)
        texts  = [d.text for d in CORPUS]
        e.fit(texts)
        chunks = [Chunk(text=d.text, doc_id=d.doc_id) for d in CORPUS]
        e.embed_chunks(chunks)
        store  = VectorStore(similarity="dot")
        store.add(chunks)
        q_emb  = e.embed("retrieval augmented generation")
        res    = store.search(q_emb, top_k=2)
        assert len(res) == 2

    def test_pipeline_dot_similarity(self):
        cfg = RAGConfig(chunk_size=200, top_k=2, embed_dim=16, similarity="dot")
        p   = RAGPipeline(cfg=cfg, embedder=TFIDFEmbedder(max_features=16))
        p.index(CORPUS)
        res = p.retrieve("NanoMind")
        assert len(res) > 0


# ── BM25 pipeline ─────────────────────────────────────────────────────────────

class TestBM25Pipeline:
    def test_bm25_pipeline_retrieves(self):
        cfg = RAGConfig(chunk_size=200, top_k=2, embed_dim=32)
        p   = RAGPipeline(cfg=cfg, embedder=BM25Embedder(max_features=32))
        p.index(CORPUS)
        res = p.retrieve("transformer attention")
        assert len(res) > 0

    def test_bm25_scores_positive(self):
        cfg = RAGConfig(chunk_size=200, top_k=3, embed_dim=32)
        p   = RAGPipeline(cfg=cfg, embedder=BM25Embedder(max_features=32))
        p.index(CORPUS)
        res = p.retrieve("NanoMind library")
        for r in res:
            assert r.score >= 0.0


# ── Deduplication ─────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_dedup_removes_duplicates(self):
        e      = TFIDFEmbedder(max_features=16)
        text   = "duplicate content here " * 3
        chunks = [Chunk(text=text, doc_id="d1"),
                  Chunk(text=text, doc_id="d2"),
                  Chunk(text="different content altogether", doc_id="d3")]
        e.fit([c.text for c in chunks])
        e.embed_chunks(chunks)
        store = VectorStore()
        store.add(chunks)
        q_emb = e.embed("duplicate content")
        res   = store.search(q_emb, top_k=3, deduplicate=True)
        texts = [r.chunk.text[:80] for r in res]
        assert len(set(texts)) == len(texts)

    def test_no_dedup_allows_duplicates(self):
        e      = TFIDFEmbedder(max_features=16)
        text   = "same text"
        chunks = [Chunk(text=text, doc_id="d1"),
                  Chunk(text=text, doc_id="d2")]
        e.fit([text])
        e.embed_chunks(chunks)
        store = VectorStore()
        store.add(chunks)
        q_emb = e.embed("same text")
        res   = store.search(q_emb, top_k=2, deduplicate=False)
        assert len(res) == 2


# ── Chunker edge cases ────────────────────────────────────────────────────────

class TestChunkerEdgeCases:
    def test_fixed_single_chunk_short_text(self):
        c      = FixedSizeChunker(chunk_size=1000, chunk_overlap=0)
        chunks = c.chunk(doc("short text"))
        assert len(chunks) == 1
        assert chunks[0].text == "short text"

    def test_fixed_no_overlap(self):
        c      = FixedSizeChunker(chunk_size=5, chunk_overlap=0)
        chunks = c.chunk(doc("abcdefghij"))
        assert chunks[0].text == "abcde"
        assert chunks[1].text == "fghij"

    def test_paragraph_single_para(self):
        c      = ParagraphChunker(chunk_size=1000)
        chunks = c.chunk(doc("Just one paragraph here."))
        assert len(chunks) == 1

    def test_chunk_preserves_doc_id(self):
        d      = doc("test content")
        c      = FixedSizeChunker(chunk_size=5, chunk_overlap=0)
        chunks = c.chunk(d)
        for ch in chunks:
            assert ch.doc_id == d.doc_id
