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


class TestRegexAutomaton:
    def test_integer_regex(self):
        reg = RegexAutomaton(r"[0-9]+")
        assert reg.accepts("12345")
        assert not reg.accepts("12a45")
        assert reg.can_accept_prefix("12")
        assert not reg.can_accept_prefix("abc")

    def test_boolean_regex(self):
        reg = RegexAutomaton(r"(true|false)")
        assert reg.accepts("true")
        assert reg.accepts("false")
        assert not reg.accepts("maybe")
        assert reg.can_accept_prefix("tr")
        assert reg.can_accept_prefix("fal")
        assert not reg.can_accept_prefix("xyz")


class TestJSONSchemaGrammar:
    def test_schema_compilation(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        grammar = JSONSchemaGrammar(schema)
        assert "root" in grammar.rules
        assert grammar.get_required_keys() == ["name", "age"]
        assert grammar.validate_instance({"name": "Alice", "age": 30}) is True
        assert grammar.validate_instance({"name": "Alice"}) is False


class TestPrefixTrie:
    def test_trie_prefix_matching(self):
        trie = PrefixTrie()
        trie.insert("cat", 0)
        trie.insert("caterpillar", 1)
        trie.insert("dog", 2)

        matches_cat = trie.find_tokens_with_prefix("cat")
        assert 0 in matches_cat
        assert 1 in matches_cat
        assert 2 not in matches_cat

        matches_do = trie.find_tokens_with_prefix("do")
        assert matches_do == [2]

        matches_none = trie.find_tokens_with_prefix("zebra")
        assert matches_none == []
