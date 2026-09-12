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
    text = re.sub(r"\*\*(.+?)\*\*", r"", text)
    text = re.sub(r"\*(.+?)\*",   r"", text)
    text = re.sub(r"```.*?```",   "",    text, flags=re.DOTALL)
    text = re.sub(r"`(.+?)`",     r"", text)
    return Document(text=text.strip(), title=p.name,
                    metadata={"source": str(p), "type": "markdown"})


def load_string(text: str, title: str = "inline", **metadata) -> Document:
    """Create a Document from a raw string."""
    return Document(text=text, title=title, metadata={"type": "string", **metadata})


def load_strings(texts: list[str], titles: list[str] | None = None) -> list[Document]:
    """Create Documents from a list of strings."""
    titles = titles or [f"doc_{i}" for i in range(len(texts))]
    return [load_string(t, title=n) for t, n in zip(texts, titles)]
