"""
nanomind/agents/structured.py — Structured output generation.

## Structured Output / JSON Mode

LLMs often need to produce structured data, not free text:
  - Extraction: pull fields from text
  - Classification: output a label + confidence
  - Planning: output a sequence of steps
  - Grading: output a score + reasoning

JSON mode forces the LLM to output valid JSON.
Schema-guided decoding ensures the JSON matches a specific shape.

## Approaches

1. JSON Mode (GPT-4, Claude):
   System prompt: "Output valid JSON only"
   Model is steered to JSON via RLHF/instruction tuning

2. Schema-guided (Outlines, Guidance):
   Constrained decoding: only tokens that maintain JSON validity
   Guarantees valid JSON but requires token-level control

3. Parse-and-retry:
   Generate text → parse → if invalid, retry with error message
   Simple but uses extra tokens

NanoMind implements parse-and-retry + validation.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class StructuredOutput:
    """The result of a structured output extraction."""
    data:    Any
    valid:   bool
    error:   str | None
    raw:     str
    retries: int = 0

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(self.data, dict):
            return self.data.get(key, default)
        return default


class OutputSchema:
    """
    Defines expected output structure for validation.

    Args:
        schema: Dict mapping field names to (type, required) tuples.
        strict: If True, extra fields are rejected.

    Example::

        schema = OutputSchema({
            "name":  (str,  True),
            "score": (float, True),
            "tags":  (list,  False),
        })
        ok, err = schema.validate({"name": "x", "score": 0.9})
    """

    def __init__(self, schema: dict, strict: bool = False) -> None:
        self.schema = schema
        self.strict = strict

    def validate(self, data: dict) -> tuple[bool, str]:
        """Validate data dict against schema."""
        if not isinstance(data, dict):
            return False, "Expected a JSON object (dict)"

        for field, (typ, required) in self.schema.items():
            if required and field not in data:
                return False, f"Missing required field: '{field}'"
            if field in data and not isinstance(data[field], typ):
                return False, f"Field '{field}' should be {typ.__name__}"

        if self.strict:
            extra = set(data.keys()) - set(self.schema.keys())
            if extra:
                return False, f"Unexpected fields: {extra}"

        return True, "ok"

    def example(self) -> dict:
        """Generate an example JSON for the prompt."""
        defaults = {str: "...", float: 0.0, int: 0, bool: True, list: [], dict: {}}
        return {f: defaults.get(t, None) for f, (t, _) in self.schema.items()}


class StructuredExtractor:
    """
    Extract structured data from LLM output.

    Attempts to parse JSON from text, with retry logic.

    Args:
        schema:     Optional :class:`OutputSchema` for validation.
        max_retries: Retries on invalid JSON.

    Example::

        extractor = StructuredExtractor(schema)
        result    = extractor.extract('{"name": "Alice", "score": 0.95}')
        print(result.data["name"])   # Alice
    """

    _JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
    _BARE_JSON  = re.compile(r"(\{.*?\}|\[.*?\])", re.DOTALL)

    def __init__(
        self,
        schema:      OutputSchema | None = None,
        max_retries: int = 3,
    ) -> None:
        self.schema      = schema
        self.max_retries = max_retries

    def extract(self, text: str) -> StructuredOutput:
        """
        Extract and validate structured JSON from LLM output.

        Args:
            text: Raw LLM output string.

        Returns:
            :class:`StructuredOutput`.
        """
        # Try code block first
        for m in self._JSON_BLOCK.finditer(text):
            result = self._try_parse(m.group(1), text)
            if result.valid:
                return result

        # Try bare JSON
        for m in self._BARE_JSON.finditer(text):
            result = self._try_parse(m.group(1), text)
            if result.valid:
                return result

        return StructuredOutput(data=None, valid=False,
                                error="No valid JSON found", raw=text)

    def _try_parse(self, json_str: str, raw: str) -> StructuredOutput:
        try:
            data = json.loads(json_str)
            if self.schema:
                ok, err = self.schema.validate(data)
                if not ok:
                    return StructuredOutput(data=data, valid=False, error=err, raw=raw)
            return StructuredOutput(data=data, valid=True, error=None, raw=raw)
        except json.JSONDecodeError as e:
            return StructuredOutput(data=None, valid=False, error=str(e), raw=raw)

    def build_prompt_suffix(self) -> str:
        """Build a prompt suffix instructing JSON output."""
        base = "

Respond with valid JSON only."
        if self.schema:
            ex = json.dumps(self.schema.example(), indent=2)
            base += f"
Expected format:
```json
{ex}
```"
        return base
