"""
examples/rag_demo.py — NanoMind RAG demo.

Indexes a corpus of NanoMind documentation and retrieves
relevant chunks for user queries.

Usage:
    python examples/rag_demo.py
"""
from nanomind.rag import (
    RAGPipeline, RAGConfig,
    load_strings,
    TFIDFEmbedder, BM25Embedder,
    FixedSizeChunker, SentenceChunker,
    build_context,
)

# ── Corpus ────────────────────────────────────────────────────────────────────
DOCS = [
    ("NanoMind Overview",
     "NanoMind is a production-grade language model library built from scratch "
     "in 30 days. It implements transformers, attention, tokenizers, LoRA, "
     "quantization, MoE, RLHF, DPO, knowledge distillation, and REST serving."),

    ("Transformer Architecture",
     "Transformers use multi-head self-attention to process sequences in parallel. "
     "Each token attends to all other tokens via query, key, and value projections. "
     "RoPE positional encoding makes attention position-aware without absolute embeddings."),

    ("Training",
     "NanoMind supports mixed-precision training (AMP), gradient checkpointing, "
     "and gradient accumulation. The trainer uses AdamW with cosine LR scheduling "
     "and supports LoRA for parameter-efficient fine-tuning."),

    ("Serving",
     "The REST API server uses Python stdlib http.server — zero dependencies. "
     "Endpoints: GET /health, GET /info, POST /generate. "
     "NanoMindClient wraps urllib for easy HTTP access."),

    ("Export",
     "NanoMind models can be exported to TorchScript (C++/mobile), "
     "ONNX (ONNX Runtime, TensorRT), and SafeTensors (HuggingFace format). "
     "INT8 dynamic quantisation reduces model size by 4x."),
]

texts  = [text for _, text in DOCS]
titles = [title for title, _ in DOCS]

# ── Build pipeline ────────────────────────────────────────────────────────────
print("=" * 60)
print("NanoMind RAG Demo")
print("=" * 60)

cfg      = RAGConfig(chunk_size=200, top_k=3, embed_dim=64)
pipeline = RAGPipeline(cfg=cfg, embedder=TFIDFEmbedder(max_features=64))
docs     = load_strings(texts, titles=titles)
pipeline.index(docs)

stats = pipeline.stats()
print(f"
Indexed: {stats['n_documents']} docs, {stats['n_chunks']} chunks")
print(f"Embedder: TF-IDF dim={stats['embed_dim']}, similarity={stats['similarity']}")

# ── Queries ───────────────────────────────────────────────────────────────────
queries = [
    "What is NanoMind?",
    "How does attention work?",
    "How do I serve the model?",
    "Can I export to ONNX?",
]

for query in queries:
    prompt, results = pipeline.query(query)
    print(f"
  Query: {query!r}")
    for r in results:
        print(f"    [{r.rank}] score={r.score:.4f}  {r.chunk.text[:80]!r}...")
    print(f"  Augmented prompt (first 200 chars): {prompt[:200]!r}")

# ── Compare TF-IDF vs BM25 ────────────────────────────────────────────────────
print("
── TF-IDF vs BM25 comparison ──")
bm25_pipeline = RAGPipeline(
    cfg=RAGConfig(chunk_size=200, top_k=2, embed_dim=64),
    embedder=BM25Embedder(max_features=64),
)
bm25_pipeline.index(docs)
q = "export model to deployment"
tfidf_res = pipeline.retrieve(q, top_k=2)
bm25_res  = bm25_pipeline.retrieve(q, top_k=2)
print(f"  Query: {q!r}")
print(f"  TF-IDF top: {tfidf_res[0].chunk.text[:60]!r}")
print(f"  BM25   top: {bm25_res[0].chunk.text[:60]!r}")
