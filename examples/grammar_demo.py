"""
examples/grammar_demo.py — End-to-end demonstration of Grammar-Constrained Decoding,
JSON Schema enforcement, and Function Calling in NanoMind.
"""
import json
import torch
from nanomind.grammar import (
    GrammarConfig,
    JSONSchemaGrammar,
    RegexAutomaton,
    LogitGrammarMasker,
    PushdownAutomaton,
    StreamingGrammarValidator,
    FunctionCallingBridge,
    GrammarEvaluator,
)


def get_weather(location: str, unit: str = "celsius", days: int = 1) -> dict:
    """Fetch weather forecast for a target location."""
    return {"location": location, "unit": unit, "days": days}


def run_demo():
    print("======================================================================")
    print("  NanoMind Day 65: Constrained Grammar & Structured Decoding")
    print("======================================================================\n")

    # [1] Function Calling Schema Extraction
    print("[1] Function Calling Schema Extraction:")
    schema = FunctionCallingBridge.function_to_schema(get_weather)
    print(f"  Target Function   : get_weather")
    print(f"  Generated Schema  :\n{json.dumps(schema, indent=2)}\n")

    # [2] JSON Schema Grammar Compilation
    print("[2] JSON Schema Grammar Compilation:")
    grammar = JSONSchemaGrammar(schema)
    required = grammar.get_required_keys()
    print(f"  Required Properties: {required}")
    print(f"  Compiled Rules     : {list(grammar.rules.keys())}\n")

    # [3] Regex Automaton Prefix Verification
    print("[3] Regex Automaton Token Filtering:")
    regex = RegexAutomaton(r"-?[0-9]+\.[0-9]+")
    prefixes = ["-", "-12", "-12.", "-12.34", "abc"]
    for p in prefixes:
        status = "ACCEPT" if regex.can_accept_prefix(p) else "REJECT"
        print(f"  Prefix '{p:8s}' -> {status}")
    print()

    # [4] Streaming Pushdown Automaton Validation
    print("[4] Streaming Pushdown Automaton (PDA):")
    pda = PushdownAutomaton()
    tokens = ['{"name":', ' "get_weather",', ' "arguments":', ' {"location":', ' "Tokyo"}}']
    for t in tokens:
        for ch in t:
            pda.step(ch)
        print(f"  Feeder Token: '{t:18s}' | Scope Stack: {pda.stack} | In String: {pda.in_string}")
    print(f"  PDA Balanced? {pda.is_balanced}\n")

    # [5] LogitGrammarMasker in Action
    print("[5] LogitGrammarMasker Simulation:")
    vocab_size = 10
    logits = torch.randn(vocab_size)
    valid_ids = [2, 5, 8]
    masker = LogitGrammarMasker(vocab_size=vocab_size)
    masked = masker.mask_logits(logits, valid_ids)
    print(f"  Original Logits: {[round(x, 2) for x in logits.tolist()[:5]]} ...")
    print(f"  Valid Token IDs: {valid_ids}")
    print(f"  Masked Allowed : {[round(masked[i].item(), 2) for i in valid_ids]}")
    print(f"  Masked Blocked : {round(masked[0].item(), 2)} (negative infinity)\n")

    # [6] Benchmark Evaluation
    print("[6] Benchmark Grammar Evaluation:")
    test_generations = [
        '{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius", "days": 3}}',
        '{"name": "get_weather", "arguments": {"location": "London", "unit": "fahrenheit", "days": 5}}',
        '{"name": "get_weather", "arguments": {"location": "Berlin", "unit": "celsius", "days": 1}}',
    ]
    eval_res = GrammarEvaluator.evaluate_generations(test_generations, schema=schema)
    print(f"  Total Generations   : {eval_res['total_samples']}")
    print(f"  Syntax Validity Rate: {eval_res['syntax_validity'] * 100}%")
    print(f"  Schema Compliance   : {eval_res['schema_compliance'] * 100}%")
    print(f"  Passed Benchmark    : {eval_res['passed_benchmark']}")

    print("\n[OK] NanoMind Constrained Grammar Decoding demo completed successfully!")


if __name__ == "__main__":
    run_demo()
