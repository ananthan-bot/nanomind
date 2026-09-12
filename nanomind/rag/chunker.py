"""
nanomind/rag/chunker.py — Text chunking strategies.

Chunking splits documents into smaller pieces that fit in the model's
context window and allow fine-grained retrieval.

Strategies:
  Fixed-size:   Split every N characters (simple, fast).
  Sentence:     Split on sentence boundaries (better coherence).
  Paragraph:    Split on blank lines (preserves structure).
  Recursive:    Try paragraph → sentence → word until target size reached.
                Used by LangChain's RecursiveCharacterTextSplitter.

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
        paragraphs = re.split(r"
\s*
", document.text.strip())
        chunks: list[Chunk] = []
        current, start_char = [], 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if sum(len(c) for c in current) + len(para) > self.chunk_size and current:
                text = "

".join(current)
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
            text = "

".join(current)
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
