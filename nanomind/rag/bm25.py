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
