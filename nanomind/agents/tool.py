"""
nanomind/agents/tool.py — Tool definition, schema, and registry.

## Function Calling in LLMs

Function calling (tool use) allows LLMs to interact with the outside world:
  - Search the web for current information
  - Call APIs (weather, stocks, calendar)
  - Execute code
  - Retrieve from databases
  - Control computer interfaces

## How it works (OpenAI-style)

1. User sends message + tool definitions (JSON schemas)
2. LLM decides whether to call a tool and generates:
   {"name": "get_weather", "arguments": {"city": "London"}}
3. Application executes the tool and returns result
4. LLM sees the result and generates final response

Tool schema (JSON Schema):
  {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      },
      "required": ["city"]
    }
  }

## Parallel Tool Calls (GPT-4-turbo, Claude)

LLMs can call multiple tools in a single response:
  [
    {"id": "call_1", "name": "search", "arguments": {"query": "..."}},
    {"id": "call_2", "name": "calculator", "arguments": {"expr": "2+2"}}
  ]
Both execute in parallel, responses added to context.

References:
  OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
  Schick et al. (2023) Toolformer: https://arxiv.org/abs/2302.04761
  Qin et al. (2023) ToolLLM: https://arxiv.org/abs/2307.16789
"""

from __future__ import annotations
import json
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolParameter:
    """A single tool parameter specification."""
    name:        str
    type:        str          # "string", "number", "boolean", "array", "object"
    description: str
    required:    bool = True
    enum:        list | None  = None
    default:     Any          = None

    def to_json_schema(self) -> dict:
        schema: dict = {"type": self.type, "description": self.description}
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class Tool:
    """
    A callable tool with JSON Schema definition.

    Args:
        name:        Tool identifier (snake_case).
        description: Human-readable description (shown to LLM).
        parameters:  List of :class:`ToolParameter`.
        func:        Callable that executes the tool.
        is_async:    Whether func is a coroutine.
        category:    Tool category (search, math, code, etc.).

    Example::

        def get_weather(city: str, unit: str = "celsius") -> str:
            return f"Weather in {city}: 22°{unit[0].upper()}"

        tool = Tool(
            name="get_weather",
            description="Get current weather for a city",
            parameters=[
                ToolParameter("city", "string", "City name"),
                ToolParameter("unit", "string", "Temperature unit",
                              required=False, enum=["celsius", "fahrenheit"]),
            ],
            func=get_weather,
        )
    """
    name:        str
    description: str
    parameters:  list[ToolParameter] = field(default_factory=list)
    func:        Callable | None     = None
    is_async:    bool                = False
    category:    str                 = "general"

    def to_schema(self) -> dict:
        """Return OpenAI-compatible tool JSON schema."""
        props    = {p.name: p.to_json_schema() for p in self.parameters}
        required = [p.name for p in self.parameters if p.required]
        return {
            "type": "function",
            "function": {
                "name":        self.name,
                "description": self.description,
                "parameters": {
                    "type":       "object",
                    "properties": props,
                    "required":   required,
                }
            }
        }

    def __call__(self, **kwargs) -> Any:
        """Execute the tool."""
        if self.func is None:
            raise RuntimeError(f"Tool '{self.name}' has no implementation.")
        return self.func(**kwargs)

    def validate_args(self, args: dict) -> tuple[bool, str]:
        """Validate arguments against schema."""
        for p in self.parameters:
            if p.required and p.name not in args:
                return False, f"Missing required parameter: '{p.name}'"
            if p.enum and p.name in args and args[p.name] not in p.enum:
                return False, f"'{p.name}' must be one of {p.enum}"
        return True, "ok"


class ToolRegistry:
    """
    Registry of available tools.

    Args:
        tools: Initial list of tools.

    Example::

        registry = ToolRegistry([weather_tool, search_tool])
        registry.register(calculator_tool)
        schemas  = registry.schemas()
        result   = registry.call("get_weather", city="London")
    """

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in (tools or []):
            self.register(t)

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def call(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found in registry.")
        ok, msg = tool.validate_args(kwargs)
        if not ok:
            raise ValueError(f"Invalid args for '{name}': {msg}")
        return tool(**kwargs)

    def schemas(self) -> list[dict]:
        """Return all tool schemas for LLM context."""
        return [t.to_schema() for t in self._tools.values()]

    def schemas_json(self) -> str:
        """Return schemas as JSON string."""
        return json.dumps(self.schemas(), indent=2)

    def by_category(self, category: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.category == category]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools.keys())


def tool(name: str | None = None, description: str = "", category: str = "general"):
    """
    Decorator to register a function as a Tool.

    Example::

        @tool(description="Add two numbers")
        def add(a: float, b: float) -> float:
            return a + b
    """
    def decorator(func: Callable) -> Tool:
        params = []
        sig    = inspect.signature(func)
        hints  = func.__annotations__
        for pname, param in sig.parameters.items():
            if pname == "return":
                continue
            py_type = hints.get(pname, str)
            type_map = {str: "string", int: "number", float: "number",
                        bool: "boolean", list: "array", dict: "object"}
            json_type = type_map.get(py_type, "string")
            required  = (param.default is inspect.Parameter.empty)
            params.append(ToolParameter(pname, json_type, f"{pname} parameter",
                                        required=required,
                                        default=None if required else param.default))
        return Tool(
            name        = name or func.__name__,
            description = description or (func.__doc__ or ""),
            parameters  = params,
            func        = func,
            category    = category,
        )
    return decorator
