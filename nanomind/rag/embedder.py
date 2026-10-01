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
