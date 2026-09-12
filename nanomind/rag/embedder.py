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
  - Use the NanoMind model's hidden states as embeddings
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
    Dense neural embedder using a NanoMind model's hidden states.

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
