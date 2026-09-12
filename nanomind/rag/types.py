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
        preview = self.text[:60].replace("
", " ")
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
        preview = self.text[:50].replace("
", " ")
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
