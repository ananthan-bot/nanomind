"""
examples/rag_demo.py — NanoMind RAG (Retrieval-Augmented Generation) demo.

Usage:
    python examples/rag_demo.py
"""
import torch
import torch.nn as nn
from nanomind.rag import (
    Chunk, FixedSizeChunker, SentenceChunker, RecursiveChunker,
    NanoEmbedder, TFIDFEmbedder, EmbeddingResult,
    VectorStore, SearchResult,
    BM25Retriever,
    HybridRetriever, CrossEncoderReranker, RetrievalResult,
    reciprocal_rank_fusion,
    RAGConfig, RAGPipeline, RAGResult,
)

V = 256

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 16)
        self.rnn  = nn.GRU(16, 32, batch_first=True)
        self.head = nn.Linear(32, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None

print("=" * 60)
print("NanoMind RAG — Retrieval-Augmented Generation Demo")
print("=" * 60)

CORPUS = [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    "Python is a high-level general-purpose programming language. It was created by Guido van Rossum.",
    "Machine learning is a subset of artificial intelligence that uses statistical techniques.",
    "The Amazon River is the largest river in the world by discharge volume of water.",
    "Photosynthesis is the process by which plants convert sunlight into glucose and oxygen.",
    "The speed of light in vacuum is approximately 299,792,458 metres per second.",
    "Napoleon Bonaparte was a French military commander who rose to prominence during the French Revolution.",
    "DNA stands for deoxyribonucleic acid and contains the genetic instructions for living organisms.",
]

# ── Chunkers ──────────────────────────────────────────────────────────────────
print("
── Chunking Strategies ──")
doc = " ".join(CORPUS)

fixed    = FixedSizeChunker(chunk_size=80, overlap=20)
sentence = SentenceChunker(max_sentences=2, overlap_sentences=1)
recursive = RecursiveChunker(chunk_size=100, overlap=20)

f_chunks = fixed.chunk(doc, "doc1")
s_chunks = sentence.chunk(doc, "doc2")
r_chunks = recursive.chunk(doc, "doc3")
print(f"  Fixed:     {len(f_chunks)} chunks, avg {sum(c.n_chars for c in f_chunks)//len(f_chunks)} chars")
print(f"  Sentence:  {len(s_chunks)} chunks")
print(f"  Recursive: {len(r_chunks)} chunks")

# ── Embedders ─────────────────────────────────────────────────────────────────
print("
── Embedders ──")
embedder = NanoEmbedder(TinyLM(), d_model=V, d_embed=32, normalise=True)
result   = embedder.encode(CORPUS[:3])
print(f"  NanoEmbedder: {tuple(result.embeddings.shape)} embeddings")
print(f"  Normalised:   {result.normalise().embeddings.norm(dim=-1).tolist()}")

tfidf = TFIDFEmbedder(vocab_size=200)
tfidf.fit(CORPUS)
tr = tfidf.encode(CORPUS[:2])
print(f"  TF-IDF:       {tuple(tr.embeddings.shape)}")

# ── Vector Store ──────────────────────────────────────────────────────────────
print("
── Vector Store ──")
chunks_for_store = [Chunk(text=t, doc_id=f"doc_{i}", chunk_id=0,
                          start_char=0, end_char=len(t))
                    for i, t in enumerate(CORPUS)]
emb_all = embedder.encode(CORPUS)
store   = VectorStore(d_embed=32)
store.add_chunks(chunks_for_store, emb_all.embeddings)
print(f"  Store stats: {store.stats()}")

query_emb = embedder.encode(["What is the tallest tower in France?"])
results   = store.search(query_emb.embeddings[0], k=3)
print(f"  Query: 'tallest tower in France'")
for r in results:
    print(f"    [{r.rank}] score={r.score:.4f}: {r.chunk.text[:60]}...")

# ── BM25 ──────────────────────────────────────────────────────────────────────
print("
── BM25 Retrieval ──")
bm25 = BM25Retriever(k1=1.5, b=0.75)
bm25.index(chunks_for_store)
bm25_res = bm25.search("Eiffel Tower Paris France", k=3)
for chunk, score in bm25_res:
    print(f"  score={score:.4f}: {chunk.text[:60]}...")

# ── Hybrid Retriever ──────────────────────────────────────────────────────────
print("
── Hybrid Retrieval (Dense + BM25 + RRF) ──")
hybrid  = HybridRetriever(store, bm25, alpha=0.5, use_rrf=True)
h_res   = hybrid.retrieve(query_emb.embeddings[0], "Eiffel Tower Paris", k=3)
for r in h_res:
    print(f"  [{r.rank}] final={r.final_score:.4f}: {r.chunk.text[:60]}...")

# RRF
list1 = chunks_for_store[:3]
list2 = chunks_for_store[1:4]
rrf   = reciprocal_rank_fusion([list1, list2], k=60)
print(f"
  RRF fusion: {len(rrf)} results merged")

# ── RAG Pipeline ──────────────────────────────────────────────────────────────
print("
── Full RAG Pipeline ──")
cfg      = RAGConfig(chunk_size=100, chunk_overlap=20, d_embed=32, top_k=3)
pipeline = RAGPipeline(embedder, TinyLM(), cfg)
for i, text in enumerate(CORPUS):
    pipeline.index_document(text, doc_id=f"fact_{i}")
print(f"  Indexed: {pipeline.stats()}")
result = pipeline.query("Where is the Eiffel Tower?")
print(f"  Question: {result.question}")
print(f"  Retrieved {result.n_chunks_retrieved} chunks")
print(f"  Top source: {result.retrieved[0].chunk.doc_id if result.retrieved else 'none'}")
print(f"  Answer: {result.answer}")
print(f"  Result dict: {result.to_dict()}")

print("
RAG demo complete!")
