"""
nanomind/grammar/regex_dfa.py — Regular expression automaton for character/token level state transitions.
"""
import re
from typing import Dict, Set, Tuple, List, Optional


class RegexAutomaton:
    """
    Finite state machine compiled from regex patterns to guide token-by-token generation.
    Supports numbers, identifiers, string literals, choices, and structured patterns.
    """
    def __init__(self, pattern: str):
        self.pattern = pattern
        self._compiled_regex = re.compile(f"^(?:{pattern})$")
        self._prefix_regex = re.compile(f"^(?:{pattern})")
        # Precomputed transition states for common regex primitives
        self.is_integer = pattern in (r"[0-9]+", r"\d+", r"-?[0-9]+")
        self.is_float = pattern in (r"[0-9]+\.[0-9]+", r"-?[0-9]+\.[0-9]+")
        self.is_boolean = pattern in (r"(true|false)", r"true|false")

    def accepts(self, text: str) -> bool:
        """Check if full string matches the regex pattern."""
        return bool(self._compiled_regex.match(text))

    def can_accept_prefix(self, prefix: str) -> bool:
        """Check if string could be a valid prefix of a matching pattern."""
        if not prefix:
            return True
        if self.accepts(prefix):
            return True
        # Check prefix compatibility
        for i in range(1, len(prefix) + 1):
            sub = prefix[:i]
            if self.is_integer:
                if not (sub.lstrip("-").isdigit() or sub == "-"):
                    return False
            elif self.is_boolean:
                if not ("true".startswith(sub) or "false".startswith(sub)):
                    return False
        return True

    def filter_valid_tokens(self, current_prefix: str, candidate_tokens: Dict[int, str]) -> List[int]:
        """Return token IDs whose addition maintains a valid prefix for the automaton."""
        valid_ids = []
        for token_id, token_str in candidate_tokens.items():
            candidate = current_prefix + token_str
            if self.can_accept_prefix(candidate):
                valid_ids.append(token_id)
        return valid_ids
