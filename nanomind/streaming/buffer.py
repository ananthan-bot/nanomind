"""
nanomind/streaming/buffer.py — Token buffer for streaming display utilities.

Provides helpers for rendering streaming output in different formats:
  - Plain print (terminal)
  - Callback-based (UI integration)
  - Batch accumulation (collect N tokens before flushing)
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from nanomind.streaming.types import StreamToken


class TokenBuffer:
    """
    Collects streaming tokens and supports multiple output modes.

    Args:
        flush_every: Flush/callback every N tokens (1 = every token).
        on_token:    Optional callback called on each flushed token.

    Example::

        buf = TokenBuffer(on_token=lambda t: print(t, end="", flush=True))
        for tok in generator.stream("Hello"):
            buf.add(tok)
        print(buf.text)
    """

    def __init__(
        self,
        flush_every: int = 1,
        on_token:    Callable[[str], None] | None = None,
    ) -> None:
        self.flush_every = flush_every
        self.on_token    = on_token
        self._tokens:    list[StreamToken] = []
        self._pending:   list[StreamToken] = []

    def add(self, token: StreamToken) -> None:
        """Add a token to the buffer."""
        self._tokens.append(token)
        self._pending.append(token)
        if len(self._pending) >= self.flush_every:
            self._flush()

    def _flush(self) -> None:
        if self._pending and self.on_token:
            text = "".join(t.token for t in self._pending)
            self.on_token(text)
        self._pending.clear()

    def drain(self) -> None:
        """Flush any remaining pending tokens."""
        self._flush()

    @property
    def text(self) -> str:
        """Full generated text so far."""
        return "".join(t.token for t in self._tokens)

    @property
    def tokens(self) -> list[StreamToken]:
        return list(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def reset(self) -> None:
        self._tokens.clear()
        self._pending.clear()


def print_stream(
    token_iter: Iterator[StreamToken],
    prefix:     str = "",
    end:        str = "
",
) -> str:
    """
    Print streaming tokens in real-time to stdout.

    Args:
        token_iter: Iterator of :class:`StreamToken`.
        prefix:     Printed before the first token.
        end:        Printed after the last token.

    Returns:
        Full generated text.
    """
    if prefix:
        sys.stdout.write(prefix)
        sys.stdout.flush()
    parts = []
    for tok in token_iter:
        sys.stdout.write(tok.token)
        sys.stdout.flush()
        parts.append(tok.token)
    sys.stdout.write(end)
    sys.stdout.flush()
    return "".join(parts)


def collect_stream(token_iter: Iterator[StreamToken]) -> tuple[str, list[StreamToken]]:
    """
    Collect all streaming tokens without printing.

    Returns:
        ``(full_text, list_of_stream_tokens)``
    """
    tokens = list(token_iter)
    return "".join(t.token for t in tokens), tokens
