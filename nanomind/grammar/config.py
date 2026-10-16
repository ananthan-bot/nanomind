"""
nanomind/grammar/config.py — Grammar representations, rules, and configuration.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any


class GrammarType(str, Enum):
    CFG = "cfg"
    REGEX = "regex"
    JSON_SCHEMA = "json_schema"
    CHOICE = "choice"


@dataclass
class GrammarRule:
    """A context-free grammar production rule."""
    name: str
    productions: List[List[str]] = field(default_factory=list)

    def add_production(self, symbols: List[str]) -> "GrammarRule":
        self.productions.append(symbols)
        return self


@dataclass
class ParseState:
    """State in a grammar parse chart or pushdown automaton."""
    rule_name: str
    prod_idx: int = 0
    dot_idx: int = 0
    stack: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.dot_idx >= len(self.stack)


@dataclass
class GrammarConfig:
    """Configuration for grammar-constrained decoding."""
    grammar_type: GrammarType = GrammarType.JSON_SCHEMA
    max_stack_depth: int = 128
    whitespace_flexible: bool = True
    allow_partial: bool = True
    strict_mode: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
