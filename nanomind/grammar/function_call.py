"""
nanomind/grammar/function_call.py — Bridge translating Python function signatures into strict grammar schemas.
"""
import inspect
import json
from typing import Callable, Dict, Any, List, Optional


class FunctionCallingBridge:
    """
    Converts native Python functions and tool declarations into JSON Schema grammars
    and parses LLM outputs directly into typed invocations.
    """
    TYPE_MAP = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    @classmethod
    def function_to_schema(cls, func: Callable) -> Dict[str, Any]:
        """Convert a Python function into an OpenAI/Anthropic compatible JSON tool schema."""
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        properties = {}
        required = []

        for name, param in sig.parameters.items():
            param_type = cls.TYPE_MAP.get(param.annotation, "string")
            properties[name] = {
                "type": param_type,
                "description": f"Parameter {name}",
            }
            if param.default == inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": [func.__name__]},
                "arguments": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
            "required": ["name", "arguments"],
        }

    @staticmethod
    def parse_invocation(json_str: str) -> Dict[str, Any]:
        """Parse structured string into function name and keyword arguments."""
        data = json.loads(json_str)
        return {
            "name": data.get("name"),
            "arguments": data.get("arguments", {}),
        }
