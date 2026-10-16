"""
tests/test_grammar.py — Comprehensive unit tests for Constrained Grammar & Structured Decoding.
"""
import json
import torch
from nanomind.grammar import (
    GrammarRule,
    GrammarConfig,
    GrammarType,
    RegexAutomaton,
    JSONSchemaGrammar,
    PrefixTrie,
    LogitGrammarMasker,
    PushdownAutomaton,
    StreamingGrammarValidator,
    FunctionCallingBridge,
    GrammarEvaluator,
)


class TestGrammarConfig:
    def test_grammar_rule_creation(self):
        rule = GrammarRule(name="expr")
        rule.add_production(["term", "+", "term"])
        rule.add_production(["term"])
        assert rule.name == "expr"
        assert len(rule.productions) == 2

    def test_grammar_config_defaults(self):
        cfg = GrammarConfig()
        assert cfg.grammar_type == GrammarType.JSON_SCHEMA
        assert cfg.max_stack_depth == 128
        assert cfg.strict_mode is True
