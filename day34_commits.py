"""
day34_commits.py — 20 atomic commits for Day 34: Retrieval-Augmented Generation (RAG).
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

print("\n=== DAY 34: Retrieval-Augmented Generation (RAG) — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — rag package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/__init__.py",
      '"""NanoMind RAG sub-package — Retrieval-Augmented Generation."""\n')
commit("feat: add nanomind/rag/ package skeleton for Retrieval-Augmented Generation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — RAGConfig
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/config.py", '''\
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
    context_template: str   = "Context:\n{context}\n\nQuestion: {query}"
    max_context_len:  int   = 2048
    deduplicate:      bool  = True

    def __post_init__(self) -> None:
        assert self.chunk_size    >  0
        assert self.chunk_overlap >= 0
        assert self.chunk_overlap <  self.chunk_size
        assert self.top_k         >= 1
        assert self.embed_dim     >= 1
        assert self.similarity in ("cosine", "dot")
''')
commit("feat: add RAGConfig — chunk_size, overlap, top_k, embed_dim, similarity, context_template")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — Document and Chunk types
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/types.py", '''\
"""
nanomind/rag/types.py — Core types for RAG: Document, Chunk, RetrievalResult.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass
class Document:
    """
    A source document to be indexed for retrieval.

    Attributes:
        text:     Full text content of the document.
        doc_id:   Unique identifier (auto-generated if not provided).
        metadata: Optional key-value metadata (source, date, author, etc.).
        title:    Optional human-readable title.
    """
    text:     str
    doc_id:   str                  = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: dict[str, Any]       = field(default_factory=dict)
    title:    str | None           = None

    def __len__(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Document(id={self.doc_id!r}, chars={len(self)}, text={preview!r}...)"


@dataclass
class Chunk:
    """
    A sub-section of a document, the unit of retrieval.

    Attributes:
        text:       Chunk text content.
        doc_id:     ID of the source document.
        chunk_id:   Unique chunk identifier.
        start_char: Character offset within source document.
        end_char:   End character offset.
        metadata:   Inherited + chunk-specific metadata.
        embedding:  Dense embedding vector (set after encoding).
    """
    text:       str
    doc_id:     str
    chunk_id:   str                = field(default_factory=lambda: str(uuid.uuid4())[:8])
    start_char: int                = 0
    end_char:   int                = 0
    metadata:   dict[str, Any]     = field(default_factory=dict)
    embedding:  list[float] | None = None

    def __len__(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        preview = self.text[:50].replace("\n", " ")
        return f"Chunk(id={self.chunk_id!r}, doc={self.doc_id!r}, text={preview!r}...)"


@dataclass
class RetrievalResult:
    """
    A single retrieved chunk with its similarity score.

    Attributes:
        chunk:  The retrieved :class:`Chunk`.
        score:  Similarity score (higher = more relevant).
        rank:   Rank in the result list (1 = most relevant).
    """
    chunk: Chunk
    score: float
    rank:  int = 1

    def __repr__(self) -> str:
        return f"RetrievalResult(rank={self.rank}, score={self.score:.4f}, chunk={self.chunk!r})"
''')
commit("feat: add Document, Chunk, RetrievalResult — core RAG data types with metadata + embeddings")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — text chunker
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/chunker.py", '''\
"""
nanomind/rag/chunker.py — Text chunking strategies.

Chunking splits documents into smaller pieces that fit in the model\'s
context window and allow fine-grained retrieval.

Strategies:
  Fixed-size:   Split every N characters (simple, fast).
  Sentence:     Split on sentence boundaries (better coherence).
  Paragraph:    Split on blank lines (preserves structure).
  Recursive:    Try paragraph → sentence → word until target size reached.
                Used by LangChain\'s RecursiveCharacterTextSplitter.

Overlap:
  Consecutive chunks overlap by `chunk_overlap` characters to avoid
  splitting context across chunk boundaries.
"""

from __future__ import annotations
import re
from nanomind.rag.types import Document, Chunk


def _make_chunks(
    text:     str,
    doc_id:   str,
    metadata: dict,
    size:     int,
    overlap:  int,
) -> list[Chunk]:
    """Slide a fixed window over text to produce overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        end   = min(start + size, len(text))
        chunk = Chunk(
            text       = text[start:end],
            doc_id     = doc_id,
            start_char = start,
            end_char   = end,
            metadata   = dict(metadata),
        )
        chunks.append(chunk)
        if end == len(text):
            break
        start += size - overlap
    return chunks


class FixedSizeChunker:
    """
    Split documents into fixed-size overlapping character chunks.

    Args:
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        return _make_chunks(
            document.text, document.doc_id, document.metadata,
            self.chunk_size, self.chunk_overlap,
        )

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        result = []
        for doc in documents:
            result.extend(self.chunk(doc))
        return result


class SentenceChunker:
    """
    Split documents on sentence boundaries, grouping into target-size chunks.

    Args:
        chunk_size:    Target maximum characters per chunk.
        chunk_overlap: Characters of overlap (in sentences) between chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self._sent_re      = re.compile(r"(?<=[.!?])\s+")

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = self._sent_re.split(document.text)
        chunks: list[Chunk] = []
        current, start_char = [], 0
        cur_len = 0

        for sent in sentences:
            if cur_len + len(sent) > self.chunk_size and current:
                text = " ".join(current)
                chunks.append(Chunk(
                    text=text, doc_id=document.doc_id,
                    start_char=start_char,
                    end_char=start_char + len(text),
                    metadata=dict(document.metadata),
                ))
                # Overlap: keep last sentence(s) within overlap budget
                kept, kept_len = [], 0
                for s in reversed(current):
                    if kept_len + len(s) <= self.chunk_overlap:
                        kept.insert(0, s); kept_len += len(s)
                    else:
                        break
                start_char = start_char + len(text) - kept_len
                current = kept; cur_len = kept_len
            current.append(sent); cur_len += len(sent)

        if current:
            text = " ".join(current)
            chunks.append(Chunk(
                text=text, doc_id=document.doc_id,
                start_char=start_char,
                end_char=start_char + len(text),
                metadata=dict(document.metadata),
            ))
        return chunks

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        result = []
        for doc in documents:
            result.extend(self.chunk(doc))
        return result


class ParagraphChunker:
    """
    Split on blank lines, merging short paragraphs to reach target size.

    Args:
        chunk_size: Target maximum characters per chunk.
    """

    def __init__(self, chunk_size: int = 512) -> None:
        self.chunk_size = chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        paragraphs = re.split(r"\n\s*\n", document.text.strip())
        chunks: list[Chunk] = []
        current, start_char = [], 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if sum(len(c) for c in current) + len(para) > self.chunk_size and current:
                text = "\n\n".join(current)
                chunks.append(Chunk(
                    text=text, doc_id=document.doc_id,
                    start_char=start_char,
                    end_char=start_char + len(text),
                    metadata=dict(document.metadata),
                ))
                start_char += len(text)
                current = []
            current.append(para)

        if current:
            text = "\n\n".join(current)
            chunks.append(Chunk(
                text=text, doc_id=document.doc_id,
                start_char=start_char,
                end_char=start_char + len(text),
                metadata=dict(document.metadata),
            ))
        return chunks

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        result = []
        for doc in documents:
            result.extend(self.chunk(doc))
        return result
''')
commit("feat: add FixedSizeChunker, SentenceChunker, ParagraphChunker — overlapping text splitting")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — TF-IDF embedder (zero-dependency)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/embedder.py", '''\
"""
nanomind/rag/embedder.py — Text embedders for RAG retrieval.

Embedding converts text into dense vectors so we can compute similarity.

## Embedder Types

TF-IDF (Term Frequency-Inverse Document Frequency):
  - Classic sparse retrieval signal, no neural network needed
  - TF:  How often a term appears in a document
  - IDF: How rare a term is across all documents (penalises stop words)
  - TF-IDF(t, d) = count(t, d) / len(d) * log(N / df(t))
  - Fast, interpretable, zero-dependency

BM25 (Best Match 25):
  - Probabilistic improvement over TF-IDF, used by Elasticsearch
  - Adds document length normalisation and saturation
  - BM25(t, d) = IDF(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))

Dense (Neural):
  - Use the NanoMind model\'s hidden states as embeddings
  - Mean-pool the last hidden layer over token positions
  - Requires a trained model but captures semantic similarity

References:
  Robertson & Zaragoza (2009) "The Probabilistic Relevance Framework: BM25"
  Karpukhin et al. (2020) "Dense Passage Retrieval" https://arxiv.org/abs/2004.04906
"""

from __future__ import annotations
import math
import re
from collections import Counter
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from nanomind.rag.types import Chunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Embedder(ABC):
    """Abstract base class for text embedders."""

    @abstractmethod
    def fit(self, texts: list[str]) -> "Embedder":
        """Fit the embedder on a corpus."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        for chunk in chunks:
            chunk.embedding = self.embed(chunk.text)
        return chunks


class TFIDFEmbedder(Embedder):
    """
    TF-IDF bag-of-words embedder (zero external dependencies).

    Args:
        max_features: Vocabulary size (most frequent terms).
        min_df:       Minimum document frequency to include a term.
    """

    def __init__(self, max_features: int = 256, min_df: int = 1) -> None:
        self.max_features = max_features
        self.min_df       = min_df
        self._vocab:   dict[str, int] = {}
        self._idf:     list[float]    = []
        self._n_docs:  int            = 0

    @property
    def embed_dim(self) -> int:
        return len(self._vocab)

    def fit(self, texts: list[str]) -> "TFIDFEmbedder":
        self._n_docs = len(texts)
        df: Counter = Counter()
        for text in texts:
            for term in set(_tokenize(text)):
                df[term] += 1
        # Filter by min_df and take top max_features
        valid = [(t, f) for t, f in df.items() if f >= self.min_df]
        valid.sort(key=lambda x: -x[1])
        vocab_terms = [t for t, _ in valid[: self.max_features]]
        self._vocab = {t: i for i, t in enumerate(vocab_terms)}
        self._idf   = [
            math.log((self._n_docs + 1) / (df[t] + 1)) + 1.0
            for t in vocab_terms
        ]
        return self

    def embed(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        tf     = Counter(tokens)
        n      = max(len(tokens), 1)
        vec    = [0.0] * len(self._vocab)
        for term, idx in self._vocab.items():
            if term in tf:
                vec[idx] = (tf[term] / n) * self._idf[idx]
        # L2 normalise
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class BM25Embedder(Embedder):
    """
    BM25 sparse embedder (zero external dependencies).

    Args:
        max_features: Vocabulary size.
        k1:           Term saturation parameter (default: 1.5).
        b:            Length normalisation parameter (default: 0.75).
    """

    def __init__(self, max_features: int = 256, k1: float = 1.5, b: float = 0.75) -> None:
        self.max_features = max_features
        self.k1           = k1
        self.b            = b
        self._vocab:   dict[str, int] = {}
        self._idf:     list[float]    = []
        self._avgdl:   float          = 1.0

    @property
    def embed_dim(self) -> int:
        return len(self._vocab)

    def fit(self, texts: list[str]) -> "BM25Embedder":
        n    = len(texts)
        df: Counter = Counter()
        lengths      = []
        for text in texts:
            toks = _tokenize(text)
            lengths.append(len(toks))
            for t in set(toks):
                df[t] += 1
        self._avgdl = sum(lengths) / max(n, 1)
        valid = sorted(df.items(), key=lambda x: -x[1])[: self.max_features]
        self._vocab = {t: i for i, (t, _) in enumerate(valid)}
        self._idf   = [
            math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1)
            for t, _ in valid
        ]
        return self

    def embed(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        tf     = Counter(tokens)
        dl     = len(tokens)
        vec    = [0.0] * len(self._vocab)
        for term, idx in self._vocab.items():
            if term in tf:
                f    = tf[term]
                norm = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                vec[idx] = self._idf[idx] * f * (self.k1 + 1) / norm
        l2 = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / l2 for v in vec]


class DenseEmbedder(Embedder):
    """
    Dense neural embedder using a NanoMind model\'s hidden states.

    Mean-pools the final hidden layer over token positions to produce
    a fixed-size dense embedding.

    Args:
        model:     NanoMind language model.
        tokenizer: Tokenizer compatible with the model.
        layer:     Which hidden layer to pool from (-1 = last).
    """

    def __init__(self, model: nn.Module, tokenizer, layer: int = -1) -> None:
        self.model     = model.eval()
        self.tokenizer = tokenizer
        self.layer     = layer

    @property
    def embed_dim(self) -> int:
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                return m.in_features
        return 128

    def fit(self, texts: list[str]) -> "DenseEmbedder":
        return self   # No fitting needed for dense embedder

    @torch.no_grad()
    def embed(self, text: str) -> list[float]:
        ids = self.tokenizer.encode(text)
        if not ids:
            return [0.0] * self.embed_dim
        x   = torch.tensor([ids[:self.model.T if hasattr(self.model, "T") else 512]])
        out, _ = self.model(x)
        vec = out[0].mean(dim=0)   # mean pool over sequence
        return vec.tolist()
''')
commit("feat: add TFIDFEmbedder, BM25Embedder, DenseEmbedder — zero-dep + neural text embedders")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — VectorStore
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/vector_store.py", '''\
"""
nanomind/rag/vector_store.py — In-memory vector store for RAG retrieval.

Stores chunk embeddings and supports fast similarity search.

## Similarity Metrics

Cosine similarity:  cos(u, v) = u·v / (|u| |v|)
  - Range: [-1, 1], 1 = identical direction
  - Scale-invariant: only direction matters, not magnitude
  - Standard for text embeddings (sentence-transformers, OpenAI)

Dot product:        u·v = Σ u_i * v_i
  - Range: (-∞, +∞), larger = more similar
  - Fast (no normalisation), used by DPR, ColBERT

## Search Complexity

Brute-force: O(N·D) per query where N=corpus size, D=embedding dim
  Acceptable for N < 100k. For larger corpora use FAISS/HNSW.

FAISS (Facebook AI Similarity Search):
  - IVF index: O(sqrt(N)·D) — partitions vectors into clusters
  - HNSW:      O(log(N)·D) — hierarchical navigable small world graph
  NanoMind implements brute-force for educational clarity.
"""

from __future__ import annotations
import math
import json
from pathlib import Path

from nanomind.rag.types import Chunk, RetrievalResult


def _cosine(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorStore:
    """
    In-memory vector store: add chunks and search by embedding similarity.

    Args:
        similarity: Similarity function — ``"cosine"`` or ``"dot"``.

    Example::

        store = VectorStore()
        store.add(chunks)          # chunks with .embedding set
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(self, similarity: str = "cosine") -> None:
        self.similarity = similarity
        self._chunks:    list[Chunk]       = []
        self._embed_dim: int | None        = None
        self._sim_fn = _cosine if similarity == "cosine" else _dot

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks (with embeddings) to the store."""
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id!r} has no embedding. "
                                 "Call embedder.embed_chunks() first.")
            if self._embed_dim is None:
                self._embed_dim = len(chunk.embedding)
            self._chunks.append(chunk)

    def search(
        self,
        query_embedding: list[float],
        top_k:           int = 5,
        deduplicate:     bool = True,
    ) -> list[RetrievalResult]:
        """
        Find the top-K most similar chunks to the query embedding.

        Args:
            query_embedding: Dense query vector.
            top_k:           Number of results to return.
            deduplicate:     Skip duplicate chunk texts.

        Returns:
            List of :class:`RetrievalResult`, sorted by score descending.
        """
        if not self._chunks:
            return []
        scores = [(self._sim_fn(query_embedding, c.embedding), c)
                  for c in self._chunks if c.embedding]
        scores.sort(key=lambda x: -x[0])

        results, seen_texts = [], set()
        for score, chunk in scores:
            if deduplicate:
                key = chunk.text.strip()[:80]
                if key in seen_texts:
                    continue
                seen_texts.add(key)
            results.append(RetrievalResult(chunk=chunk, score=score, rank=len(results)+1))
            if len(results) >= top_k:
                break
        return results

    def __len__(self) -> int:
        return len(self._chunks)

    def save(self, path: str | Path) -> None:
        """Save store to a JSON file."""
        data = [{
            "text":       c.text,
            "doc_id":     c.doc_id,
            "chunk_id":   c.chunk_id,
            "start_char": c.start_char,
            "end_char":   c.end_char,
            "metadata":   c.metadata,
            "embedding":  c.embedding,
        } for c in self._chunks]
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: str | Path) -> "VectorStore":
        """Load store from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._chunks = [Chunk(**d) for d in data]
        if self._chunks and self._chunks[0].embedding:
            self._embed_dim = len(self._chunks[0].embedding)
        return self
''')
commit("feat: add VectorStore — cosine/dot similarity search, add/search/save/load, deduplication")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — context builder
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/context.py", '''\
"""
nanomind/rag/context.py — Build augmented prompts from retrieved chunks.
"""

from __future__ import annotations
from nanomind.rag.types import RetrievalResult


def build_context(
    results:        list[RetrievalResult],
    max_chars:      int = 2048,
    separator:      str = "\n\n---\n\n",
    show_scores:    bool = False,
    show_sources:   bool = True,
) -> str:
    """
    Format retrieved chunks into a context string for the prompt.

    Args:
        results:      Retrieved chunks with scores.
        max_chars:    Maximum total characters in the context.
        separator:    String between consecutive chunks.
        show_scores:  Prepend similarity scores to each chunk.
        show_sources: Prepend doc_id source labels.

    Returns:
        Formatted context string.
    """
    parts:  list[str] = []
    total   = 0
    for r in results:
        header  = ""
        if show_sources:
            header += f"[Source: {r.chunk.doc_id}]"
        if show_scores:
            header += f" [Score: {r.score:.4f}]"
        part = (header + "\n" + r.chunk.text).strip() if header else r.chunk.text
        if total + len(part) > max_chars:
            # Truncate last chunk to fit
            remaining = max_chars - total
            if remaining > 100:
                parts.append(part[:remaining])
            break
        parts.append(part)
        total += len(part) + len(separator)
    return separator.join(parts)


def build_rag_prompt(
    query:          str,
    results:        list[RetrievalResult],
    template:       str = "Context:\n{context}\n\nQuestion: {query}\nAnswer:",
    max_context:    int = 2048,
    show_sources:   bool = True,
) -> str:
    """
    Build a complete RAG-augmented prompt.

    Args:
        query:        User query string.
        results:      Retrieved :class:`RetrievalResult` list.
        template:     Prompt template with ``{context}`` and ``{query}`` placeholders.
        max_context:  Max context characters.
        show_sources: Include source document IDs in context.

    Returns:
        Full augmented prompt string.
    """
    context = build_context(results, max_chars=max_context, show_sources=show_sources)
    return template.format(context=context, query=query)
''')
commit("feat: add build_context(), build_rag_prompt() — format retrieved chunks into LLM prompts")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — RAGPipeline (unified)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/pipeline.py", '''\
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
''')
commit("feat: add RAGPipeline — index(), retrieve(), augment(), query(), stats() end-to-end pipeline")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — document loaders
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/loaders.py", '''\
"""
nanomind/rag/loaders.py — Document loaders for common file formats.
"""

from __future__ import annotations
from pathlib import Path
from nanomind.rag.types import Document


def load_text_file(path: str | Path, encoding: str = "utf-8") -> Document:
    """Load a plain text file as a Document."""
    p    = Path(path)
    text = p.read_text(encoding=encoding)
    return Document(text=text, title=p.name, metadata={"source": str(p), "type": "text"})


def load_text_files(directory: str | Path, pattern: str = "*.txt") -> list[Document]:
    """Load all text files matching a glob pattern from a directory."""
    return [load_text_file(p) for p in sorted(Path(directory).glob(pattern))]


def load_markdown_file(path: str | Path, encoding: str = "utf-8") -> Document:
    """Load a Markdown file, stripping common Markdown syntax."""
    import re
    p    = Path(path)
    text = p.read_text(encoding=encoding)
    # Strip Markdown headers, bold, italic, code fences
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*",   r"\1", text)
    text = re.sub(r"```.*?```",   "",    text, flags=re.DOTALL)
    text = re.sub(r"`(.+?)`",     r"\1", text)
    return Document(text=text.strip(), title=p.name,
                    metadata={"source": str(p), "type": "markdown"})


def load_string(text: str, title: str = "inline", **metadata) -> Document:
    """Create a Document from a raw string."""
    return Document(text=text, title=title, metadata={"type": "string", **metadata})


def load_strings(texts: list[str], titles: list[str] | None = None) -> list[Document]:
    """Create Documents from a list of strings."""
    titles = titles or [f"doc_{i}" for i in range(len(texts))]
    return [load_string(t, title=n) for t, n in zip(texts, titles)]
''')
commit("feat: add loaders — load_text_file, load_markdown_file, load_string, load_strings")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — rag __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/rag/__init__.py", '''\
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
''')
commit("refactor: export all RAG components from nanomind/rag/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — example: rag_demo.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/rag_demo.py", '''\
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
print(f"\nIndexed: {stats['n_documents']} docs, {stats['n_chunks']} chunks")
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
    print(f"\n  Query: {query!r}")
    for r in results:
        print(f"    [{r.rank}] score={r.score:.4f}  {r.chunk.text[:80]!r}...")
    print(f"  Augmented prompt (first 200 chars): {prompt[:200]!r}")

# ── Compare TF-IDF vs BM25 ────────────────────────────────────────────────────
print("\n── TF-IDF vs BM25 comparison ──")
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
''')
commit("feat: add examples/rag_demo.py — index/retrieve/augment demo, TF-IDF vs BM25 comparison")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12-19 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_rag.py", '''\
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
        text = "Para one content.\n\nPara two content.\n\nPara three."
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
''')
commit("test: add full RAG test suite — config, types, chunkers, embedders, store, context, pipeline")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — load_strings test
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_rag.py")
src += '''

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
        f.write_text("# Title\n\nSome content.", encoding="utf-8")
        from nanomind.rag import load_markdown_file
        d = load_markdown_file(f)
        assert "#" not in d.text
        assert "Title" in d.text
'''
write("tests/test_rag.py", src)
commit("test: add loader tests — load_string, load_strings, load_text_file, load_markdown strips headers")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — dot similarity test
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_rag.py")
src += '''

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
'''
write("tests/test_rag.py", src)
commit("test: add dot similarity VectorStore and RAGPipeline dot metric tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — BM25 pipeline test
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_rag.py")
src += '''

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
'''
write("tests/test_rag.py", src)
commit("test: add BM25 pipeline retrieves and positive score tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — deduplication test
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_rag.py")
src += '''

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
'''
write("tests/test_rag.py", src)
commit("test: add deduplication removes duplicates and no-dedup allows duplicates tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — sentence chunker edge cases
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_rag.py")
src += '''

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
'''
write("tests/test_rag.py", src)
commit("test: add chunker edge cases — single chunk, no overlap, single paragraph, doc_id preserved")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — VectorStore save/load roundtrip
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_rag.py")
src += '''

# ── VectorStore persistence ───────────────────────────────────────────────────

class TestVectorStorePersistence:
    def test_search_after_load(self, tmp_path):
        e      = TFIDFEmbedder(max_features=16)
        e.fit([d.text for d in CORPUS])
        chunks = [Chunk(text=d.text, doc_id=d.doc_id) for d in CORPUS]
        e.embed_chunks(chunks)
        store  = VectorStore()
        store.add(chunks)
        path   = tmp_path / "vs.json"
        store.save(path)

        store2 = VectorStore()
        store2.load(path)
        q_emb  = e.embed("NanoMind")
        res    = store2.search(q_emb, top_k=2)
        assert len(res) == 2

    def test_save_json_valid(self, tmp_path):
        e      = TFIDFEmbedder(max_features=8)
        e.fit(["hello world"])
        chunk  = Chunk(text="hello", doc_id="d1")
        e.embed_chunks([chunk])
        store  = VectorStore()
        store.add([chunk])
        path   = tmp_path / "vs.json"
        store.save(path)
        data   = json.loads(path.read_text())
        assert isinstance(data, list)
        assert "embedding" in data[0]
'''
write("tests/test_rag.py", src)
commit("test: add VectorStore save/load search roundtrip and valid JSON output tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v3.4.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"3.3.0\"", "__version__ = \"3.4.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v3.4.0 — Retrieval-Augmented Generation (RAG) release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `prompt` | Prompt Templates — ChatML, LLaMA2/3, Alpaca, few-shot, PromptManager |",
    "| `prompt` | Prompt Templates — ChatML, LLaMA2/3, Alpaca, few-shot, PromptManager |\n"
    "| `rag`    | RAG — chunking, TF-IDF/BM25/dense embedders, VectorStore, pipeline |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = "## [3.4.0] — 2024 — Retrieval-Augmented Generation (RAG)\n\n### Added\n" \
     "- `RAGPipeline` — end-to-end index/retrieve/augment/query/stats\n" \
     "- `RAGConfig` — chunk_size, top_k, embed_dim, similarity, context_template\n" \
     "- `Document`, `Chunk`, `RetrievalResult` — core RAG data types\n" \
     "- `FixedSizeChunker` / `SentenceChunker` / `ParagraphChunker` — text splitting\n" \
     "- `TFIDFEmbedder` — TF-IDF sparse embedder (zero external deps)\n" \
     "- `BM25Embedder` — BM25 probabilistic sparse embedder\n" \
     "- `DenseEmbedder` — NanoMind neural mean-pool dense embedder\n" \
     "- `VectorStore` — cosine/dot similarity brute-force search + save/load JSON\n" \
     "- `build_context()` / `build_rag_prompt()` — context injection into prompts\n" \
     "- `load_text_file()` / `load_markdown_file()` / `load_string()` — document loaders\n" \
     "- `examples/rag_demo.py` — full pipeline + TF-IDF vs BM25 comparison\n\n---\n\n" + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v3.4.0, update README and CHANGELOG for Day 34 RAG")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 34 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v3.4.0",
    "-m", "NanoMind v3.4.0 — Retrieval-Augmented Generation", check=False)
r = run("git", "push", "origin", "v3.4.0", check=False)
print("Tag v3.4.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 34 COMPLETE — v3.4.0 TAGGED! ===")
