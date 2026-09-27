"""
nanomind/agents/builtin_tools.py — Built-in demo tools for NanoMind agents.

These tools demonstrate the tool use framework with simple,
self-contained implementations (no external APIs needed):

  - calculator:  Arithmetic expression evaluator
  - word_count:  Count words in text
  - reverse:     Reverse a string
  - upper/lower: String case conversion
  - get_time:    Current timestamp
  - web_search:  Mock web search (returns canned results)
  - memory:      In-process key-value store

All tools work without network access for offline development.
"""

from __future__ import annotations
import ast
import operator
import time
from datetime import datetime
from nanomind.agents.tool import Tool, ToolParameter, ToolRegistry


# ── Calculator ────────────────────────────────────────────────────────────────

def _safe_eval(expr: str) -> float:
    """Safely evaluate arithmetic expressions."""
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.n if hasattr(node, 'n') else node.value
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        else:
            raise ValueError("Unsupported expression")
    return float(_eval(ast.parse(expr, mode="eval").body))


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        result = _safe_eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def word_count(text: str) -> str:
    """Count the number of words, sentences, and characters in text."""
    words     = len(text.split())
    sentences = text.count(".") + text.count("!") + text.count("?")
    chars     = len(text)
    return f"Words: {words}, Sentences: {sentences}, Characters: {chars}"


def reverse_string(text: str) -> str:
    """Reverse the characters in a string."""
    return text[::-1]


def to_uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def get_datetime() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")


def mock_search(query: str, n_results: int = 3) -> str:
    """Mock web search — returns simulated search results."""
    results = [
        f"[Result {i+1}] {query} — simulated search result {i+1}. "
        f"This is a mock result for development purposes."
        for i in range(min(n_results, 5))
    ]
    return "
".join(results)


# In-memory key-value store
_memory_store: dict[str, str] = {}

def memory_set(key: str, value: str) -> str:
    """Store a key-value pair in memory."""
    _memory_store[key] = value
    return f"Stored: {key} = {value}"

def memory_get(key: str) -> str:
    """Retrieve a value from memory by key."""
    if key in _memory_store:
        return f"{key} = {_memory_store[key]}"
    return f"Key '{key}' not found in memory."


# ── Tool definitions ──────────────────────────────────────────────────────────

CALCULATOR_TOOL = Tool(
    name="calculator",
    description="Evaluate a mathematical expression (+, -, *, /, **)",
    parameters=[ToolParameter("expression", "string", "Math expression to evaluate")],
    func=calculator,
    category="math",
)

WORD_COUNT_TOOL = Tool(
    name="word_count",
    description="Count words, sentences, and characters in text",
    parameters=[ToolParameter("text", "string", "Text to analyse")],
    func=word_count,
    category="text",
)

REVERSE_TOOL = Tool(
    name="reverse_string",
    description="Reverse the characters in a string",
    parameters=[ToolParameter("text", "string", "Text to reverse")],
    func=reverse_string,
    category="text",
)

UPPERCASE_TOOL = Tool(
    name="to_uppercase",
    description="Convert text to uppercase",
    parameters=[ToolParameter("text", "string", "Text to convert")],
    func=to_uppercase,
    category="text",
)

DATETIME_TOOL = Tool(
    name="get_datetime",
    description="Get the current date and time",
    parameters=[],
    func=lambda: get_datetime(),
    category="utility",
)

SEARCH_TOOL = Tool(
    name="web_search",
    description="Search the web for information (mock implementation)",
    parameters=[
        ToolParameter("query",     "string", "Search query"),
        ToolParameter("n_results", "number", "Number of results", required=False, default=3),
    ],
    func=mock_search,
    category="search",
)

MEMORY_SET_TOOL = Tool(
    name="memory_set",
    description="Store a key-value pair in agent memory",
    parameters=[
        ToolParameter("key",   "string", "Memory key"),
        ToolParameter("value", "string", "Value to store"),
    ],
    func=memory_set,
    category="memory",
)

MEMORY_GET_TOOL = Tool(
    name="memory_get",
    description="Retrieve a value from agent memory",
    parameters=[ToolParameter("key", "string", "Memory key to retrieve")],
    func=memory_get,
    category="memory",
)


def default_registry() -> ToolRegistry:
    """Create a registry with all built-in demo tools."""
    return ToolRegistry([
        CALCULATOR_TOOL, WORD_COUNT_TOOL, REVERSE_TOOL,
        UPPERCASE_TOOL, DATETIME_TOOL, SEARCH_TOOL,
        MEMORY_SET_TOOL, MEMORY_GET_TOOL,
    ])
