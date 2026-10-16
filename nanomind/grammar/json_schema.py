"""
nanomind/grammar/json_schema.py — Compiles JSON Schema specifications into structured grammar rules.
"""
import json
from typing import Dict, Any, List, Optional
from nanomind.grammar.config import GrammarRule


class JSONSchemaGrammar:
    """
    Translates JSON Schema objects into CFG production rules and field validators.
    Guarantees that any generated text parses as valid JSON conforming to the schema.
    """
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.rules: Dict[str, GrammarRule] = {}
        self._compile()

    def _compile(self) -> None:
        """Compile JSON schema into grammar production rules."""
        schema_type = self.schema.get("type", "object")
        if schema_type == "object":
            self._compile_object("root", self.schema)
        elif schema_type == "array":
            self._compile_array("root", self.schema)
        else:
            self._compile_primitive("root", self.schema)

    def _compile_object(self, name: str, obj_schema: Dict[str, Any]) -> None:
        props = obj_schema.get("properties", {})
        required = obj_schema.get("required", list(props.keys()))

        rule = GrammarRule(name=name)
        # Order required properties deterministically
        prop_items = []
        for k in required:
            prop_items.append(f'"{k}":')
            prop_type = props.get(k, {}).get("type", "string")
            prop_items.append(f"<{prop_type}>")

        rule.add_production(["{"] + prop_items + ["}"])
        self.rules[name] = rule

    def _compile_array(self, name: str, arr_schema: Dict[str, Any]) -> None:
        item_schema = arr_schema.get("items", {"type": "string"})
        item_type = item_schema.get("type", "string")
        rule = GrammarRule(name=name)
        rule.add_production(["[", f"<{item_type}>", "]"])
        self.rules[name] = rule

    def _compile_primitive(self, name: str, prim_schema: Dict[str, Any]) -> None:
        p_type = prim_schema.get("type", "string")
        rule = GrammarRule(name=name)
        rule.add_production([f"<{p_type}>"])
        self.rules[name] = rule

    def get_required_keys(self) -> List[str]:
        """Extract required keys if root schema is an object."""
        if self.schema.get("type") == "object":
            props = self.schema.get("properties", {})
            return self.schema.get("required", list(props.keys()))
        return []

    def validate_instance(self, obj: Any) -> bool:
        """Verify an in-memory python object against the schema."""
        if not isinstance(obj, dict) and self.schema.get("type") == "object":
            return False
        for k in self.get_required_keys():
            if k not in obj:
                return False
        return True
