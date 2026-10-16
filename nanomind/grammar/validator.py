"""
nanomind/grammar/validator.py — Streaming pushdown automaton and structural validation for partial outputs.
"""
import json
from typing import List, Tuple, Dict, Any, Optional


class PushdownAutomaton:
    """Stack-based pushdown automaton tracking nested scopes: braces, brackets, and strings."""
    def __init__(self):
        self.stack: List[str] = []
        self.in_string: bool = False
        self.escape_next: bool = False

    def step(self, ch: str) -> bool:
        """Advance the automaton with a single character. Returns False on syntax error."""
        if self.in_string:
            if self.escape_next:
                self.escape_next = False
                return True
            if ch == "\\":
                self.escape_next = True
                return True
            if ch == '"':
                self.in_string = False
                return True
            return True

        if ch == '"':
            self.in_string = True
            return True
        elif ch in ("{", "["):
            self.stack.append(ch)
            return True
        elif ch == "}":
            if not self.stack or self.stack[-1] != "{":
                return False
            self.stack.pop()
            return True
        elif ch == "]":
            if not self.stack or self.stack[-1] != "[":
                return False
            self.stack.pop()
            return True
        return True

    @property
    def is_balanced(self) -> bool:
        return len(self.stack) == 0 and not self.in_string


class StreamingGrammarValidator:
    """
    Validates partial streaming text token-by-token against structured JSON rules.
    Allows early termination when parsing irrecoverably fails.
    """
    def __init__(self):
        self.pda = PushdownAutomaton()
        self.buffer = ""

    def feed_token(self, token: str) -> bool:
        """Consume a token string. Returns True if prefix is syntactically sound."""
        self.buffer += token
        for ch in token:
            if not self.pda.step(ch):
                return False
        return True

    def is_complete_json(self) -> bool:
        """Check if accumulated buffer forms a complete, valid JSON object."""
        if not self.pda.is_balanced or not self.buffer.strip():
            return False
        try:
            json.loads(self.buffer)
            return True
        except ValueError:
            return False

    def get_parsed_json(self) -> Optional[Any]:
        """Return parsed JSON object if complete, else None."""
        try:
            return json.loads(self.buffer)
        except ValueError:
            return None
