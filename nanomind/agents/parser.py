"""
nanomind/agents/parser.py — Parse LLM output into structured tool calls.

LLMs generate tool calls as text. This module parses them into
structured :class:`ToolCall` objects.

Supported formats:
  1. JSON block (OpenAI style):
     {"name": "search", "arguments": {"query": "..."}}

  2. XML tags (Anthropic Claude style):
     <tool_call><name>search</name><arguments>{"query": "..."}</arguments></tool_call>

  3. Markdown code block:
     ```json
     {"name": "search", "arguments": {"query": "..."}}
     ```

  4. ReAct format (Yao et al., 2022):
     Action: search
     Action Input: {"query": "latest news"}
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A parsed tool call from LLM output."""
    id:        str
    name:      str
    arguments: dict
    raw:       str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @staticmethod
    def make_id(n: int = 0) -> str:
        import time
        return f"call_{n}_{int(time.time() * 1000) % 100000}"


@dataclass
class ToolResult:
    """Result from executing a tool call."""
    call_id:  str
    name:     str
    result:   str
    error:    str | None = None
    success:  bool       = True

    def to_message(self) -> dict:
        """Convert to message dict for LLM context."""
        return {
            "role":         "tool",
            "tool_call_id": self.call_id,
            "name":         self.name,
            "content":      self.result if self.success else f"Error: {self.error}",
        }


class ToolCallParser:
    """
    Parse LLM text output into :class:`ToolCall` objects.

    Supports multiple output formats for flexibility.

    Example::

        parser = ToolCallParser()
        text   = '{"name": "search", "arguments": {"query": "Paris weather"}}'
        calls  = parser.parse(text)
        # → [ToolCall(name="search", arguments={"query": "Paris weather"})]
    """

    # JSON in code block
    _CODE_BLOCK  = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    # Bare JSON object
    _BARE_JSON   = re.compile(r"\{[^{}]*"name"\s*:\s*"[^"]+"[^{}]*\}", re.DOTALL)
    # XML style
    _XML_CALL    = re.compile(
        r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE
    )
    # ReAct style
    _REACT_ACT   = re.compile(r"Action\s*:\s*(\w+)\s*
Action Input\s*:\s*(.+?)(?=
|$)",
                               re.DOTALL)

    def parse(self, text: str) -> list[ToolCall]:
        """
        Parse all tool calls from LLM output.

        Tries formats in order: code block → XML → bare JSON → ReAct.

        Args:
            text: LLM output string.

        Returns:
            List of :class:`ToolCall` objects.
        """
        calls: list[ToolCall] = []

        # 1. Code blocks
        for i, m in enumerate(self._CODE_BLOCK.finditer(text)):
            tc = self._parse_json(m.group(1), i)
            if tc:
                calls.append(tc)

        if not calls:
            # 2. XML
            for i, m in enumerate(self._XML_CALL.finditer(text)):
                tc = self._parse_xml_content(m.group(1), i)
                if tc:
                    calls.append(tc)

        if not calls:
            # 3. Bare JSON
            for i, m in enumerate(self._BARE_JSON.finditer(text)):
                tc = self._parse_json(m.group(0), i)
                if tc:
                    calls.append(tc)

        if not calls:
            # 4. ReAct
            for i, m in enumerate(self._REACT_ACT.finditer(text)):
                tc = self._parse_react(m.group(1), m.group(2), i)
                if tc:
                    calls.append(tc)

        return calls

    def _parse_json(self, text: str, idx: int) -> ToolCall | None:
        try:
            d = json.loads(text.strip())
            name = d.get("name") or d.get("function", {}).get("name")
            args = d.get("arguments") or d.get("parameters") or {}
            if name:
                return ToolCall(
                    id=ToolCall.make_id(idx), name=name,
                    arguments=args if isinstance(args, dict) else json.loads(args),
                    raw=text
                )
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def _parse_xml_content(self, content: str, idx: int) -> ToolCall | None:
        name_m = re.search(r"<name>(.*?)</name>", content)
        args_m = re.search(r"<arguments>(.*?)</arguments>", content, re.DOTALL)
        if name_m:
            name = name_m.group(1).strip()
            args = {}
            if args_m:
                try:
                    args = json.loads(args_m.group(1).strip())
                except json.JSONDecodeError:
                    args = {"raw": args_m.group(1).strip()}
            return ToolCall(id=ToolCall.make_id(idx), name=name, arguments=args)
        return None

    def _parse_react(self, action: str, action_input: str, idx: int) -> ToolCall | None:
        name = action.strip()
        try:
            args = json.loads(action_input.strip())
        except json.JSONDecodeError:
            args = {"input": action_input.strip()}
        return ToolCall(id=ToolCall.make_id(idx), name=name, arguments=args)

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains any tool call."""
        return bool(self.parse(text))

    def extract_final_answer(self, text: str) -> str | None:
        """Extract final answer from ReAct-style output."""
        m = re.search(r"Final Answer\s*:\s*(.+?)$", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None
