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

    DEFAULT_SEPARATORS = ["

", "
", ". ", " ", ""]

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
