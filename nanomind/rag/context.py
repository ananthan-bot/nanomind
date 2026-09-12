"""
nanomind/rag/context.py — Build augmented prompts from retrieved chunks.
"""

from __future__ import annotations
from nanomind.rag.types import RetrievalResult


def build_context(
    results:        list[RetrievalResult],
    max_chars:      int = 2048,
    separator:      str = "

---

",
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
        part = (header + "
" + r.chunk.text).strip() if header else r.chunk.text
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
    template:       str = "Context:
{context}

Question: {query}
Answer:",
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
