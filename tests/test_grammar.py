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


class TestLogitGrammarMasker:
    def test_logit_masking_1d(self):
        masker = LogitGrammarMasker(vocab_size=5, fill_value=-1e9)
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        valid_ids = [1, 3]
        masked = masker.mask_logits(logits, valid_ids)

        assert masked[1] == 2.0
        assert masked[3] == 4.0
        assert masked[0] == -1e9
        assert masked[2] == -1e9
        assert masked[4] == -1e9

    def test_boolean_mask_tensor(self):
        masker = LogitGrammarMasker(vocab_size=4)
        mask = masker.create_mask_tensor([0, 2])
        assert mask.tolist() == [True, False, True, False]


class TestStreamingGrammarValidator:
    def test_pushdown_automaton_balance(self):
        pda = PushdownAutomaton()
        s = '{"key": [1, 2, "hello"]}'
        for ch in s:
            assert pda.step(ch) is True
        assert pda.is_balanced is True

    def test_streaming_validator_completion(self):
        val = StreamingGrammarValidator()
        assert not val.is_complete_json()
        val.feed_token('{"user":')
        val.feed_token(' "Alice"}')
        assert val.is_complete_json() is True
        assert val.get_parsed_json() == {"user": "Alice"}


class TestFunctionCallingBridge:
    def test_function_schema_generation(self):
        def sample_tool(city: str, radius: int = 10) -> bool:
            return True

        schema = FunctionCallingBridge.function_to_schema(sample_tool)
        assert schema["type"] == "object"
        assert "properties" in schema["properties"]["arguments"]
        assert "city" in schema["properties"]["arguments"]["required"]

    def test_evaluator_metrics(self):
        res = GrammarEvaluator.evaluate_generations(
            generations=['{"a": 1}', '{"a": 2}'],
            schema={"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
        )
        assert res["syntax_validity"] == 1.0
        assert res["schema_compliance"] == 1.0
        assert res["passed_benchmark"] is True
