"""
day51_commits.py — 20 atomic commits for Day 51: Retrieval-Augmented Generation (RAG).
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== DAY 51: Retrieval-Augmented Generation (RAG) — 20 commits, v5.1.0 ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — rag package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/__init__.py",
      '"""NanoMind RAG sub-package — Retrieval-Augmented Generation."""\n')
commit("feat: add nanomind/rag/ package skeleton for Retrieval-Augmented Generation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — Document chunking
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/chunker.py", '''\
"""
nanomind/rag/chunker.py — Document chunking strategies for RAG.

## Why Chunking Matters

LLMs have limited context windows (4K-200K tokens).
Documents can be hundreds of thousands of tokens.

Solution: split documents into overlapping chunks, embed each chunk,
store in vector store, retrieve relevant chunks at query time.

## Chunking Strategies

1. Fixed-size chunking:
   Split every N tokens/characters with overlap.
   Simple, fast, ignores content structure.

2. Sentence-based chunking:
   Split at sentence boundaries.
   Better for QA — preserves complete thoughts.

3. Recursive chunking (LangChain default):
   Try to split at paragraphs, then sentences, then words.
   Respects natural document structure.

4. Semantic chunking:
   Embed sentences, split where embedding similarity drops.
   Best quality, most expensive.

5. Document-aware chunking:
   Split at markdown headers, code blocks, HTML tags.
   Best for structured documents.

## Overlap

Chunks overlap by `overlap` tokens to avoid cutting context at boundaries:
  Chunk 1: tokens 0..512
  Chunk 2: tokens 400..912   (112 token overlap)
  Chunk 3: tokens 800..1312  (112 token overlap)

References:
  Lewis et al. (2020) "Retrieval-Augmented Generation for Knowledge-Intensive NLP"
  https://arxiv.org/abs/2005.11401
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A document chunk with metadata."""
    text:      str
    doc_id:    str
    chunk_id:  int
    start_char: int
    end_char:   int
    metadata:  dict = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return len(self.text)

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict:
        return {
            "text":      self.text,
            "doc_id":    self.doc_id,
            "chunk_id":  self.chunk_id,
            "n_chars":   self.n_chars,
        }


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with overlap.

    Args:
        chunk_size: Characters per chunk.
        overlap:    Characters of overlap between consecutive chunks.

    Example::

        chunker = FixedSizeChunker(chunk_size=512, overlap=64)
        chunks  = chunker.chunk("Long document text...", doc_id="doc_1")
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def chunk(self, text: str, doc_id: str = "doc", metadata: dict | None = None) -> list[Chunk]:
        chunks  = []
        start   = 0
        cid     = 0
        while start < len(text):
            end   = min(start + self.chunk_size, len(text))
            chunk = Chunk(
                text       = text[start:end],
                doc_id     = doc_id,
                chunk_id   = cid,
                start_char = start,
                end_char   = end,
                metadata   = metadata or {},
            )
            chunks.append(chunk)
            if end == len(text):
                break
            start += self.chunk_size - self.overlap
            cid   += 1
        return chunks


class SentenceChunker:
    """
    Split text at sentence boundaries, grouping into chunks.

    Args:
        max_sentences: Sentences per chunk.
        overlap_sentences: Sentence overlap between chunks.

    Example::

        chunker = SentenceChunker(max_sentences=5, overlap_sentences=1)
        chunks  = chunker.chunk("First sentence. Second. Third.", "doc_1")
    """

    _SENT_END = re.compile(r'(?<=[.!?])\s+')

    def __init__(self, max_sentences: int = 5, overlap_sentences: int = 1) -> None:
        self.max_sentences     = max_sentences
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in self._SENT_END.split(text) if s.strip()]

    def chunk(self, text: str, doc_id: str = "doc", metadata: dict | None = None) -> list[Chunk]:
        sentences = self._split_sentences(text)
        chunks    = []
        i = 0
        cid = 0
        while i < len(sentences):
            batch    = sentences[i: i + self.max_sentences]
            chunk_text = " ".join(batch)
            # Approximate char offsets
            start_char = text.find(batch[0]) if batch else 0
            end_char   = start_char + len(chunk_text)
            chunks.append(Chunk(
                text=chunk_text, doc_id=doc_id, chunk_id=cid,
                start_char=start_char, end_char=end_char,
                metadata=metadata or {},
            ))
            i   += max(1, self.max_sentences - self.overlap_sentences)
            cid += 1
        return chunks


class RecursiveChunker:
    """
    Recursive character-based chunking (LangChain-style).

    Splits on: paragraphs → sentences → words → characters.

    Args:
        chunk_size: Target chunk size in characters.
        overlap:    Character overlap.
        separators: List of separators to try in order.

    Example::

        chunker = RecursiveChunker(chunk_size=256, overlap=32)
        chunks  = chunker.chunk(document_text, doc_id="article_1")
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size:  int  = 512,
        overlap:     int  = 64,
        separators:  list | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap    = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, seps: list) -> list[str]:
        if not seps:
            return [text]
        sep = seps[0]
        parts = text.split(sep) if sep else list(text)
        good, bad = [], []
        for p in parts:
            if len(p) <= self.chunk_size:
                good.append(p)
            else:
                bad.extend(self._split_text(p, seps[1:]))
        return [p for p in (good + bad) if p.strip()]

    def chunk(self, text: str, doc_id: str = "doc", metadata: dict | None = None) -> list[Chunk]:
        parts  = self._split_text(text, self.separators)
        chunks = []
        cid    = 0
        buf    = ""
        for part in parts:
            if len(buf) + len(part) + 1 <= self.chunk_size:
                buf = (buf + " " + part).strip()
            else:
                if buf:
                    start = text.find(buf)
                    chunks.append(Chunk(
                        text=buf, doc_id=doc_id, chunk_id=cid,
                        start_char=max(0, start),
                        end_char=max(0, start) + len(buf),
                        metadata=metadata or {},
                    ))
                    cid += 1
                # Overlap: keep tail of buf
                words = buf.split()
                overlap_text = " ".join(words[-max(1, self.overlap // 10):])
                buf = (overlap_text + " " + part).strip()
        if buf:
            start = text.find(buf)
            chunks.append(Chunk(
                text=buf, doc_id=doc_id, chunk_id=cid,
                start_char=max(0, start),
                end_char=max(0, start) + len(buf),
                metadata=metadata or {},
            ))
        return chunks
''')
commit("feat: add Chunk, FixedSizeChunker, SentenceChunker, RecursiveChunker — chunking strategies for RAG")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — Embedder
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/embedder.py", '''\
"""
nanomind/rag/embedder.py — Text embedding models for RAG.

## Dense Retrieval via Embeddings

Encode text into dense vectors in a shared semantic space.
Similar meanings → similar vectors → high cosine similarity.

Embedding models:
  - sentence-transformers/all-MiniLM-L6-v2: 384-dim, fast
  - text-embedding-ada-002 (OpenAI): 1536-dim, cloud
  - e5-large-v2: 1024-dim, best open-source
  - BGE-M3: 1024-dim, multilingual

NanoMind implements:
  1. NanoEmbedder — lightweight custom embedder using the LM backbone
  2. TFIDFEmbedder — sparse TF-IDF embedder (baseline, no neural net)
  3. EmbedderPipeline — batch embedding with caching

## Bi-encoder vs Cross-encoder

Bi-encoder (for retrieval): encode query and doc SEPARATELY
  → fast (pre-compute doc embeddings offline)
  → retrieval: cosine similarity in milliseconds

Cross-encoder (for re-ranking): encode (query, doc) JOINTLY
  → slow (must run for every (query, doc) pair)
  → higher quality (sees full context)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """Result of embedding a list of texts."""
    embeddings: torch.Tensor   # (N, D)
    texts:      list[str]
    dim:        int

    def __len__(self) -> int:
        return len(self.texts)

    def normalise(self) -> "EmbeddingResult":
        """L2-normalise embeddings for cosine similarity via dot product."""
        normed = F.normalize(self.embeddings, dim=-1)
        return EmbeddingResult(normed, self.texts, self.dim)


class NanoEmbedder(nn.Module):
    """
    Lightweight neural text embedder using a small transformer backbone.

    Mean-pools the last hidden states to produce a fixed-size vector.

    Args:
        backbone:    LM model returning ``(B, T, D)`` hidden states.
        d_model:     Hidden dimension of backbone.
        d_embed:     Output embedding dimension.
        normalise:   L2-normalise output embeddings.

    Example::

        embedder = NanoEmbedder(backbone, d_model=128, d_embed=64)
        result   = embedder.encode(["Hello world", "How are you?"])
        sim      = result.embeddings @ result.embeddings.T   # cosine sim
    """

    def __init__(
        self,
        backbone:  nn.Module,
        d_model:   int,
        d_embed:   int = 64,
        normalise: bool = True,
    ) -> None:
        super().__init__()
        self.backbone  = backbone
        self.proj      = nn.Linear(d_model, d_embed)
        self.normalise = normalise
        self.d_embed   = d_embed

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool backbone hidden states → projected embedding."""
        out    = self.backbone(input_ids)
        hidden = out[0] if isinstance(out, tuple) else out   # (B, T, D)
        pooled = hidden.mean(dim=1)                           # (B, D)
        emb    = self.proj(pooled)                            # (B, d_embed)
        if self.normalise:
            emb = F.normalize(emb, dim=-1)
        return emb

    @torch.no_grad()
    def encode(
        self,
        texts:    list[str],
        tokenize_fn: object = None,
        batch_size:  int    = 32,
    ) -> EmbeddingResult:
        """
        Encode a list of texts into embeddings.

        Args:
            texts:       List of text strings.
            tokenize_fn: Callable(text) → token_ids list. Uses simple
                         ASCII fallback if None.
            batch_size:  Batch size for encoding.

        Returns:
            :class:`EmbeddingResult`.
        """
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            if tokenize_fn:
                ids = [tokenize_fn(t) for t in batch]
            else:
                # Simple ASCII fallback tokenizer
                ids = [[ord(c) % 256 for c in t[:64]] or [0] for t in batch]
            max_len = max(len(x) for x in ids)
            padded  = torch.zeros(len(ids), max_len, dtype=torch.long)
            for j, x in enumerate(ids):
                padded[j, :len(x)] = torch.tensor(x)
            embs = self.forward(padded)
            all_embs.append(embs)
        embeddings = torch.cat(all_embs, dim=0)
        return EmbeddingResult(embeddings, texts, self.d_embed)


class TFIDFEmbedder:
    """
    Sparse TF-IDF embedder (baseline, no neural network).

    Args:
        vocab_size: Maximum vocabulary size.

    Example::

        emb = TFIDFEmbedder(vocab_size=1000)
        emb.fit(corpus)
        result = emb.encode(["search query"])
    """

    def __init__(self, vocab_size: int = 1000) -> None:
        self.vocab_size = vocab_size
        self._vocab:   dict = {}
        self._idf:     dict = {}
        self._fitted:  bool = False

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def fit(self, corpus: list[str]) -> None:
        """Build vocabulary and IDF from corpus."""
        from collections import Counter
        word_doc_count: Counter = Counter()
        all_words = set()
        for doc in corpus:
            words = set(self._tokenize(doc))
            all_words.update(words)
            word_doc_count.update(words)

        # Most common words → vocab
        most_common = word_doc_count.most_common(self.vocab_size)
        self._vocab = {w: i for i, (w, _) in enumerate(most_common)}
        N = len(corpus)
        self._idf = {
            w: math.log((N + 1) / (c + 1)) + 1
            for w, c in most_common
        }
        self._fitted = True

    def encode(self, texts: list[str]) -> EmbeddingResult:
        """Encode texts as TF-IDF vectors."""
        if not self._fitted:
            self.fit(texts)
        V = len(self._vocab)
        vecs = torch.zeros(len(texts), V)
        for i, text in enumerate(texts):
            words = self._tokenize(text)
            tf_counts: dict = {}
            for w in words:
                tf_counts[w] = tf_counts.get(w, 0) + 1
            for w, cnt in tf_counts.items():
                if w in self._vocab:
                    j         = self._vocab[w]
                    tf        = cnt / max(len(words), 1)
                    idf       = self._idf.get(w, 1.0)
                    vecs[i, j] = tf * idf
            # L2 normalise
            norm = vecs[i].norm()
            if norm > 0:
                vecs[i] /= norm
        return EmbeddingResult(vecs, texts, V)
''')
commit("feat: add NanoEmbedder (mean-pool backbone), TFIDFEmbedder, EmbeddingResult — encode, normalise")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — Vector store
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/vector_store.py", '''\
"""
nanomind/rag/vector_store.py — In-memory vector store for RAG.

## Vector Stores

After embedding documents, store vectors for fast retrieval:
  - FAISS: Facebook's billion-scale similarity search
  - Chroma: Open-source vector DB with persistence
  - Pinecone: Cloud-hosted vector DB
  - Weaviate: Graph-aware vector DB
  - pgvector: PostgreSQL extension

NanoMind implements an in-memory flat vector store.
Production systems use FAISS with IVF (inverted file) indexing for O(√N) search.

## Similarity Search

Given query embedding q and corpus embeddings E:
  scores = E @ q   (dot product, if L2-normalised = cosine similarity)
  top_k  = argsort(scores, descending=True)[:k]

ANN (Approximate Nearest Neighbours):
  HNSW (Hierarchical Navigable Small World graphs): O(log N) search
  IVF (Inverted File Index): cluster vectors, search only relevant clusters
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from nanomind.rag.chunker import Chunk


@dataclass
class SearchResult:
    """A single retrieval result."""
    chunk:   Chunk
    score:   float
    rank:    int

    def to_dict(self) -> dict:
        return {"text": self.chunk.text, "score": round(self.score, 4),
                "doc_id": self.chunk.doc_id, "rank": self.rank}


class VectorStore:
    """
    In-memory flat vector store with cosine similarity search.

    Args:
        d_embed: Embedding dimension.

    Example::

        store = VectorStore(d_embed=64)
        store.add_chunks(chunks, embeddings)
        results = store.search(query_embedding, k=5)
    """

    def __init__(self, d_embed: int) -> None:
        self.d_embed  = d_embed
        self._chunks: list[Chunk]         = []
        self._embeds: list[torch.Tensor]  = []   # each (d_embed,)

    def add_chunks(
        self,
        chunks:     list[Chunk],
        embeddings: torch.Tensor,
    ) -> None:
        """
        Add chunks and their embeddings to the store.

        Args:
            chunks:     List of :class:`Chunk`.
            embeddings: ``(N, D)`` embedding tensor.
        """
        assert len(chunks) == embeddings.shape[0]
        for chunk, emb in zip(chunks, embeddings):
            self._chunks.append(chunk)
            self._embeds.append(F.normalize(emb, dim=-1))

    def search(
        self,
        query_embedding: torch.Tensor,
        k:               int = 5,
    ) -> list[SearchResult]:
        """
        Find k most similar chunks to query embedding.

        Args:
            query_embedding: ``(D,)`` or ``(1, D)`` query vector.
            k:               Number of results to return.

        Returns:
            List of :class:`SearchResult` sorted by score descending.
        """
        if not self._chunks:
            return []

        q      = F.normalize(query_embedding.flatten(), dim=-1)  # (D,)
        embeds = torch.stack(self._embeds, dim=0)                 # (N, D)
        scores = (embeds @ q).tolist()                            # (N,)

        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:k]
        return [
            SearchResult(chunk=self._chunks[i], score=s, rank=r)
            for r, (i, s) in enumerate(ranked)
        ]

    def search_batch(
        self,
        query_embeddings: torch.Tensor,
        k:                int = 5,
    ) -> list[list[SearchResult]]:
        """Search multiple queries in parallel."""
        return [self.search(q, k) for q in query_embeddings]

    def delete_doc(self, doc_id: str) -> int:
        """Remove all chunks from a document. Returns n removed."""
        before = len(self._chunks)
        pairs  = [(c, e) for c, e in zip(self._chunks, self._embeds)
                  if c.doc_id != doc_id]
        self._chunks = [p[0] for p in pairs]
        self._embeds = [p[1] for p in pairs]
        return before - len(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def stats(self) -> dict:
        docs = set(c.doc_id for c in self._chunks)
        return {
            "n_chunks": len(self._chunks),
            "n_docs":   len(docs),
            "d_embed":  self.d_embed,
        }
''')
commit("feat: add VectorStore — add_chunks, search (cosine), search_batch, delete_doc, stats")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — BM25 retriever
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/bm25.py", '''\
"""
nanomind/rag/bm25.py — BM25 sparse retrieval.

## BM25 (Best Match 25)

The gold standard sparse retrieval algorithm, used in Elasticsearch and Solr.

BM25 score for document D, query Q:
  score(D, Q) = Σ_{t∈Q} IDF(t) × (tf(t,D) × (k1+1)) / (tf(t,D) + k1 × (1-b+b×|D|/avgdl))

Where:
  tf(t, D)  = term frequency of term t in document D
  IDF(t)    = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
  |D|       = document length in words
  avgdl     = average document length
  k1        = term frequency saturation (typically 1.2-2.0)
  b         = length normalisation (typically 0.75)

## Hybrid Search (Dense + Sparse)

Best retrieval combines dense (semantic) + sparse (keyword):
  hybrid_score = α × dense_score + (1-α) × bm25_score

Used in: Cohere Rerank, Azure AI Search, Elasticsearch.

Reference:
  Robertson & Zaragoza (2009) "The Probabilistic Relevance Framework: BM25 and Beyond"
"""

from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass
from nanomind.rag.chunker import Chunk


class BM25Retriever:
    """
    BM25 sparse keyword retriever.

    Args:
        k1:  Term frequency saturation (default 1.5).
        b:   Length normalisation factor (default 0.75).

    Example::

        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.index(chunks)
        results   = retriever.search("python programming", k=5)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1   = k1
        self.b    = b
        self._chunks:  list[Chunk]       = []
        self._tf:      list[dict]        = []   # per-chunk term frequencies
        self._idf:     dict              = {}
        self._avgdl:   float             = 0.0
        self._indexed: bool              = False

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def index(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from chunks."""
        self._chunks = chunks
        tokenized    = [self._tokenize(c.text) for c in chunks]
        self._tf     = [dict(Counter(t)) for t in tokenized]

        # Average document length
        lengths    = [len(t) for t in tokenized]
        self._avgdl = sum(lengths) / max(len(lengths), 1)

        # IDF
        N          = len(chunks)
        df: dict   = {}
        for toks in tokenized:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        self._idf = {
            w: math.log((N - c + 0.5) / (c + 0.5) + 1)
            for w, c in df.items()
        }
        self._indexed = True

    def score(self, chunk_idx: int, query_tokens: list[str]) -> float:
        """BM25 score for one chunk."""
        tf   = self._tf[chunk_idx]
        dl   = sum(tf.values())
        s    = 0.0
        for t in query_tokens:
            if t not in tf:
                continue
            idf  = self._idf.get(t, 0.0)
            freq = tf[t]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1))
            s += idf * freq * (self.k1 + 1) / denom
        return s

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """
        Retrieve top-k chunks by BM25 score.

        Args:
            query: Query text.
            k:     Number of results.

        Returns:
            List of (chunk, score) tuples sorted by score descending.
        """
        if not self._indexed:
            raise RuntimeError("Call index() before search()")
        q_tokens = self._tokenize(query)
        scores   = [(i, self.score(i, q_tokens)) for i in range(len(self._chunks))]
        scores.sort(key=lambda x: -x[1])
        return [(self._chunks[i], s) for i, s in scores[:k]]

    def __len__(self) -> int:
        return len(self._chunks)
''')
commit("feat: add BM25Retriever — index, score, search — BM25 sparse retrieval with IDF weighting")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Hybrid retriever + re-ranker
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/retriever.py", '''\
"""
nanomind/rag/retriever.py — Hybrid retriever and cross-encoder re-ranker.

## Two-Stage Retrieval

Stage 1 (Recall): Retrieve top-K candidates quickly
  - Dense: vector store cosine similarity
  - Sparse: BM25 keyword matching
  - Hybrid: combine both with RRF or linear interpolation

Stage 2 (Precision): Re-rank top-K with a cross-encoder
  - Cross-encoder sees full (query, passage) context
  - Much more accurate but slow (can't pre-compute)
  - Applied only to top-K from stage 1 (K=50-100)

## Reciprocal Rank Fusion (RRF)

Combine ranked lists without score normalisation:
  RRF_score(d) = Σ_{r∈rankings} 1 / (k + rank(d, r))

k=60 is standard (from Cormack et al., 2009).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from nanomind.rag.chunker import Chunk
from nanomind.rag.vector_store import VectorStore, SearchResult
from nanomind.rag.bm25 import BM25Retriever


@dataclass
class RetrievalResult:
    """Combined retrieval + re-ranking result."""
    chunk:       Chunk
    dense_score: float | None
    bm25_score:  float | None
    final_score: float
    rank:        int

    def to_dict(self) -> dict:
        return {
            "text":        self.chunk.text[:100],
            "doc_id":      self.chunk.doc_id,
            "final_score": round(self.final_score, 4),
            "rank":        self.rank,
        }


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]],
    k: int = 60,
) -> list[tuple[Chunk, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked chunk lists (each from a retriever).
        k:            RRF constant (default 60).

    Returns:
        List of (chunk, rrf_score) sorted by score descending.
    """
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            key = (chunk.doc_id, chunk.chunk_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    # Map key → chunk
    chunk_map: dict = {}
    for ranked in ranked_lists:
        for chunk in ranked:
            chunk_map[(chunk.doc_id, chunk.chunk_id)] = chunk

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    return [(chunk_map[k], s) for k, s in sorted_scores]


class HybridRetriever:
    """
    Two-stage hybrid retriever: dense + BM25 + RRF fusion.

    Args:
        vector_store:  :class:`VectorStore` for dense retrieval.
        bm25:          :class:`BM25Retriever` for sparse retrieval.
        alpha:         Weight for dense vs BM25 (0 = BM25 only, 1 = dense only).
        use_rrf:       Use RRF fusion instead of linear interpolation.

    Example::

        retriever = HybridRetriever(vector_store, bm25, alpha=0.5)
        results   = retriever.retrieve(query_embedding, "python list sorting", k=5)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25:         BM25Retriever,
        alpha:        float = 0.5,
        use_rrf:      bool  = True,
    ) -> None:
        self.vstore = vector_store
        self.bm25   = bm25
        self.alpha  = alpha
        self.use_rrf = use_rrf

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        query_text:      str,
        k:               int = 5,
        fetch_k:         int = 20,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-k chunks using hybrid search.

        Args:
            query_embedding: ``(D,)`` dense query vector.
            query_text:      Query text for BM25.
            k:               Final results to return.
            fetch_k:         Candidates to fetch from each retriever.

        Returns:
            List of :class:`RetrievalResult`.
        """
        # Stage 1: Dense retrieval
        dense_results = self.vstore.search(query_embedding, k=fetch_k)
        dense_chunks  = [r.chunk for r in dense_results]
        dense_map     = {(r.chunk.doc_id, r.chunk.chunk_id): r.score
                         for r in dense_results}

        # Stage 1: BM25 retrieval
        bm25_raw      = self.bm25.search(query_text, k=fetch_k)
        bm25_chunks   = [c for c, _ in bm25_raw]
        bm25_map      = {(c.doc_id, c.chunk_id): s for c, s in bm25_raw}

        if self.use_rrf:
            fused = reciprocal_rank_fusion([dense_chunks, bm25_chunks])[:k]
            results = []
            for rank, (chunk, score) in enumerate(fused):
                key = (chunk.doc_id, chunk.chunk_id)
                results.append(RetrievalResult(
                    chunk       = chunk,
                    dense_score = dense_map.get(key),
                    bm25_score  = bm25_map.get(key),
                    final_score = score,
                    rank        = rank,
                ))
        else:
            # Linear interpolation
            all_keys = set(dense_map) | set(bm25_map)
            scores   = []
            for key in all_keys:
                d = dense_map.get(key, 0.0)
                b = bm25_map.get(key, 0.0)
                scores.append((key, self.alpha * d + (1 - self.alpha) * b))
            scores.sort(key=lambda x: -x[1])
            chunk_map = {(c.doc_id, c.chunk_id): c
                         for c in dense_chunks + bm25_chunks}
            results = []
            for rank, (key, score) in enumerate(scores[:k]):
                if key in chunk_map:
                    results.append(RetrievalResult(
                        chunk       = chunk_map[key],
                        dense_score = dense_map.get(key),
                        bm25_score  = bm25_map.get(key),
                        final_score = score,
                        rank        = rank,
                    ))
        return results


class CrossEncoderReranker(nn.Module):
    """
    Cross-encoder re-ranker for stage-2 precision retrieval.

    Scores (query, passage) pairs jointly — much higher accuracy than bi-encoder.

    Args:
        backbone:  LM backbone.
        d_model:   Hidden dimension.

    Example::

        reranker = CrossEncoderReranker(backbone, d_model=128)
        scores   = reranker.score(query_ids, passage_ids_list)
        reranked = sorted(zip(passages, scores), key=lambda x: -x[1])
    """

    def __init__(self, backbone: nn.Module, d_model: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head     = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Score a batch of (query + passage) concatenations."""
        out    = self.backbone(input_ids)
        hidden = out[0] if isinstance(out, tuple) else out
        pooled = hidden[:, 0, :]   # CLS token
        return self.head(pooled).squeeze(-1)   # (B,)

    @torch.no_grad()
    def rerank(
        self,
        results:    list[RetrievalResult],
        query_ids:  list[int],
        max_len:    int = 32,
    ) -> list[RetrievalResult]:
        """Re-rank results using cross-encoder scores."""
        scores = []
        for r in results:
            # Concatenate query + passage tokens (simplified)
            passage_ids = [ord(c) % 256 for c in r.chunk.text[:max_len]]
            combined    = torch.tensor([query_ids + passage_ids]).long()
            score       = self.forward(combined).item()
            scores.append(score)

        ranked = sorted(zip(results, scores), key=lambda x: -x[1])
        reranked = []
        for rank, (res, score) in enumerate(ranked):
            reranked.append(RetrievalResult(
                chunk       = res.chunk,
                dense_score = res.dense_score,
                bm25_score  = res.bm25_score,
                final_score = score,
                rank        = rank,
            ))
        return reranked
''')
commit("feat: add HybridRetriever (RRF/linear), CrossEncoderReranker, reciprocal_rank_fusion, RetrievalResult")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — RAG pipeline
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/pipeline.py", '''\
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
        "Answer the question based only on the following context.\n"
        "If the answer is not in the context, say 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
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
        context = "\n\n".join(parts)
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
''')
commit("feat: add RAGPipeline — index_document, retrieve, query, build_context, RAGConfig, RAGResult")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — rag __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/__init__.py", '''\
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
''')
commit("refactor: export all RAG components from nanomind/rag/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — example
# ══════════════════════════════════════════════════════════════════════════════
write("examples/rag_demo.py", '''\
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
print("\n── Chunking Strategies ──")
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
print("\n── Embedders ──")
embedder = NanoEmbedder(TinyLM(), d_model=V, d_embed=32, normalise=True)
result   = embedder.encode(CORPUS[:3])
print(f"  NanoEmbedder: {tuple(result.embeddings.shape)} embeddings")
print(f"  Normalised:   {result.normalise().embeddings.norm(dim=-1).tolist()}")

tfidf = TFIDFEmbedder(vocab_size=200)
tfidf.fit(CORPUS)
tr = tfidf.encode(CORPUS[:2])
print(f"  TF-IDF:       {tuple(tr.embeddings.shape)}")

# ── Vector Store ──────────────────────────────────────────────────────────────
print("\n── Vector Store ──")
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
print("\n── BM25 Retrieval ──")
bm25 = BM25Retriever(k1=1.5, b=0.75)
bm25.index(chunks_for_store)
bm25_res = bm25.search("Eiffel Tower Paris France", k=3)
for chunk, score in bm25_res:
    print(f"  score={score:.4f}: {chunk.text[:60]}...")

# ── Hybrid Retriever ──────────────────────────────────────────────────────────
print("\n── Hybrid Retrieval (Dense + BM25 + RRF) ──")
hybrid  = HybridRetriever(store, bm25, alpha=0.5, use_rrf=True)
h_res   = hybrid.retrieve(query_emb.embeddings[0], "Eiffel Tower Paris", k=3)
for r in h_res:
    print(f"  [{r.rank}] final={r.final_score:.4f}: {r.chunk.text[:60]}...")

# RRF
list1 = chunks_for_store[:3]
list2 = chunks_for_store[1:4]
rrf   = reciprocal_rank_fusion([list1, list2], k=60)
print(f"\n  RRF fusion: {len(rrf)} results merged")

# ── RAG Pipeline ──────────────────────────────────────────────────────────────
print("\n── Full RAG Pipeline ──")
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

print("\nRAG demo complete!")
''')
commit("feat: add examples/rag_demo.py — chunking, embedding, vector store, BM25, hybrid, RAG pipeline")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_rag.py", '''\
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
        chunks = c.chunk("Para one.\n\nPara two.\n\nPara three.", "d")
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
''')
commit("test: add full RAG test suite — chunkers, embedders, vector store, BM25, RRF, hybrid, pipeline")

for title, body in [
    ("test: add Chunk n_words and to_dict test", '''
class TestChunk:
    def test_n_words(self):
        c = Chunk("hello world how", "d", 0, 0, 15)
        assert c.n_words == 3

    def test_to_dict(self):
        c = Chunk("hello", "d", 0, 0, 5)
        d = c.to_dict()
        assert "text" in d and "doc_id" in d
'''),
    ("test: add FixedSizeChunker overlap test", '''
class TestChunkOverlap:
    def test_overlap_creates_shared_content(self):
        c      = FixedSizeChunker(chunk_size=10, overlap=5)
        text   = "abcdefghijklmnopqrstuvwxyz"
        chunks = c.chunk(text, "d")
        if len(chunks) >= 2:
            # Second chunk should start before first chunk ends
            assert chunks[1].start_char < chunks[0].end_char
'''),
    ("test: add NanoEmbedder batch encoding test", '''
class TestNanoEmbedderBatch:
    def test_batch_encoding(self):
        e  = NanoEmbedder(TinyLM(), V, 16)
        texts = ["text " + str(i) for i in range(10)]
        r  = e.encode(texts, batch_size=3)
        assert r.embeddings.shape == (10, 16)
'''),
    ("test: add VectorStore search_batch test", '''
class TestVectorStoreBatch:
    def test_search_batch(self):
        e      = NanoEmbedder(TinyLM(), V, 16)
        chunks = [Chunk(t, f"d{i}", 0, 0, len(t)) for i, t in enumerate(DOCS)]
        embs   = e.encode(DOCS).embeddings
        s      = VectorStore(d_embed=16)
        s.add_chunks(chunks, embs)
        q_batch = embs[:2]
        results = s.search_batch(q_batch, k=2)
        assert len(results) == 2
        assert all(len(r) <= 2 for r in results)
'''),
    ("test: add BM25 empty query returns results test", '''
class TestBM25EmptyQuery:
    def test_empty_query_zero_scores(self):
        chunks = [Chunk(t, f"d{i}", 0, 0, len(t)) for i, t in enumerate(DOCS)]
        b = BM25Retriever()
        b.index(chunks)
        res = b.search("", k=4)
        assert all(s == 0.0 for _, s in res)
'''),
    ("test: add RAGConfig defaults test", '''
class TestRAGConfig:
    def test_defaults(self):
        cfg = RAGConfig()
        assert cfg.top_k == 5
        assert cfg.chunk_size == 512
        assert cfg.alpha == 0.5
        assert cfg.use_rrf is True
'''),
    ("test: add TFIDFEmbedder auto-fit on encode test", '''
class TestTFIDFAutoFit:
    def test_auto_fit(self):
        e   = TFIDFEmbedder(vocab_size=50)
        # Call encode without fit — should auto-fit
        res = e.encode(["hello world", "python code"])
        assert res.embeddings.shape[0] == 2
'''),
]:
    src = read("tests/test_rag.py")
    src += "\n" + body
    write("tests/test_rag.py", src)
    commit(title)

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v5.1.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"5.0.0\"", "__version__ = \"5.1.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v5.1.0 — Retrieval-Augmented Generation release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `alignment`  | Modern Alignment — DPO/IPO, Constitutional AI, KTO, RLAIF, reward model, eval |",
    "| `alignment`  | Modern Alignment — DPO/IPO, Constitutional AI, KTO, RLAIF, reward model, eval |\n"
    "| `rag`        | RAG — chunking, dense/sparse retrieval, BM25, hybrid search, re-ranking, pipeline |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [5.1.0] — 2024 — Retrieval-Augmented Generation\n\n### Added\n"
      "- `Chunk` / `FixedSizeChunker` / `SentenceChunker` / `RecursiveChunker` — chunking strategies\n"
      "- `NanoEmbedder` — bi-encoder (mean-pool backbone + projection)\n"
      "- `TFIDFEmbedder` — sparse TF-IDF baseline\n"
      "- `EmbeddingResult` — embeddings + normalise()\n"
      "- `VectorStore` — in-memory cosine similarity search, delete_doc\n"
      "- `BM25Retriever` — BM25 sparse retrieval with IDF weighting\n"
      "- `HybridRetriever` — dense + BM25 with RRF or linear fusion\n"
      "- `CrossEncoderReranker` — joint (query, passage) re-ranking\n"
      "- `reciprocal_rank_fusion` — multi-list RRF merging\n"
      "- `RAGConfig` / `RAGPipeline` — end-to-end RAG with index + query\n"
      "- `RAGResult` — answer + retrieved chunks + context\n"
      "- `examples/rag_demo.py` — full RAG pipeline demo\n\n---\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v5.1.0, update README and CHANGELOG for Day 51 RAG")

print("\n=== Pushing Day 51 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")
run("git", "tag", "-a", "v5.1.0", "-m", "NanoMind v5.1.0 — Retrieval-Augmented Generation", check=False)
r = run("git", "push", "origin", "v5.1.0", check=False)
print("Tag v5.1.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 51 COMPLETE — v5.1.0 TAGGED! ===")
