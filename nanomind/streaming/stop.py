"""
nanomind/streaming/stop.py — Stop sequence detection for streaming generation.

Detects when the model has generated a stop sequence (e.g. "<|endoftext|>",
"\n\n", "</s>") and halts generation early.

The challenge: a stop sequence may span multiple tokens.
  e.g. stop="</s>" might be tokenised as ["<", "/", "s", ">"]
  We must check the decoded running text, not individual tokens.
"""

from __future__ import annotations


class StopSequenceDetector:
    """
    Detect stop sequences in streaming token output.

    Args:
        stop_sequences: List of strings that trigger early stopping.

    Example::

        detector = StopSequenceDetector(["\n\n", "<|end|>"])
        for token_str in tokens:
            detector.update(token_str)
            if detector.should_stop():
                break
        text = detector.get_text()  # text up to (not including) stop sequence
    """

    def __init__(self, stop_sequences: list[str]) -> None:
        self.stops   = [s for s in stop_sequences if s]
        self._buffer = ""
        self._stop_found: str | None = None

    def update(self, token: str) -> bool:
        """
        Add a token to the buffer and check for stop sequences.

        Returns:
            True if a stop sequence was found.
        """
        self._buffer += token
        for stop in self.stops:
            if stop in self._buffer:
                self._stop_found = stop
                return True
        return False

    def should_stop(self) -> bool:
        """Return True if any stop sequence has been detected."""
        return self._stop_found is not None

    def get_text(self) -> str:
        """Return generated text up to (not including) the stop sequence."""
        if self._stop_found:
            idx = self._buffer.find(self._stop_found)
            return self._buffer[:idx]
        return self._buffer

    def reset(self) -> None:
        """Reset state for a new generation."""
        self._buffer      = ""
        self._stop_found  = None

    @property
    def generated_text(self) -> str:
        return self._buffer

    @property
    def stop_reason(self) -> str | None:
        return self._stop_found
