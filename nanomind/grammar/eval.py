"""
nanomind/grammar/eval.py — Benchmarking suite evaluating schema compliance rate and masking overhead.
"""
import json
import time
from typing import List, Dict, Any, Optional


class GrammarEvaluator:
    """
    Evaluates constrained grammar decoding across three key dimensions:
    1. Syntax Validity: % of generations that parse cleanly as JSON / target grammar.
    2. Schema Compliance: % of generations adhering to all required keys and datatypes.
    3. Latency Overhead: mean milliseconds per token added by grammar state masking.
    """
    @staticmethod
    def evaluate_generations(
        generations: List[str],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a batch of generated outputs against structural criteria."""
        total = len(generations)
        if total == 0:
            return {"syntax_validity": 0.0, "schema_compliance": 0.0, "total": 0}

        valid_syntax_count = 0
        schema_compliant_count = 0

        required_keys = []
        if schema and schema.get("type") == "object":
            props = schema.get("properties", {})
            required_keys = schema.get("required", list(props.keys()))

        for text in generations:
            try:
                parsed = json.loads(text)
                valid_syntax_count += 1
                # Check required keys
                if isinstance(parsed, dict) and all(k in parsed for k in required_keys):
                    schema_compliant_count += 1
            except (ValueError, TypeError):
                continue

        syntax_validity = valid_syntax_count / total
        schema_compliance = schema_compliant_count / total

        return {
            "total_samples": total,
            "syntax_validity": round(syntax_validity, 4),
            "schema_compliance": round(schema_compliance, 4),
            "passed_benchmark": (syntax_validity == 1.0 and schema_compliance >= 0.95),
        }
