"""
day47_commits.py — 20 atomic commits for Day 47: Tool Use & Agentic LLMs.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== DAY 47: Tool Use & Agentic LLMs — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — agents package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/agents/__init__.py",
      '"""NanoMind Agents sub-package — Tool Use & Agentic LLM infrastructure."""\n')
commit("feat: add nanomind/agents/ package skeleton for tool use and agentic LLMs")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — Tool registry and schema
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/agents/tool.py", '''\
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
''')
commit("feat: add Tool, ToolParameter, ToolRegistry, @tool decorator — JSON schema, call, validate_args")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — Tool call parsing and structured output
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/agents/parser.py", '''\
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
    _BARE_JSON   = re.compile(r"\{[^{}]*\"name\"\s*:\s*\"[^\"]+\"[^{}]*\}", re.DOTALL)
    # XML style
    _XML_CALL    = re.compile(
        r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE
    )
    # ReAct style
    _REACT_ACT   = re.compile(r"Action\s*:\s*(\w+)\s*\nAction Input\s*:\s*(.+?)(?=\n|$)",
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
''')
commit("feat: add ToolCall, ToolResult, ToolCallParser — JSON/XML/ReAct format parsing, extract_final_answer")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — ReAct agent
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/agents/react.py", '''\
"""
nanomind/agents/react.py — ReAct: Reasoning + Acting agent loop.

## ReAct (Yao et al., 2022)

ReAct interleaves reasoning traces and actions:

  Thought: I need to find the population of France.
  Action: search
  Action Input: {"query": "population of France 2024"}
  Observation: France has 68 million people as of 2024.
  Thought: Now I have the answer.
  Final Answer: France has approximately 68 million people.

This trace-then-act pattern gives several benefits:
  ✓ Transparent reasoning (interpretable)
  ✓ Grounded answers (facts from tools)
  ✓ Error recovery (can retry on observation)
  ✓ Long-horizon tasks (multi-step)

## System Prompt for ReAct

The LLM needs a system prompt explaining the format:
  "You have access to these tools: {tools}
   Think step by step using Thought: / Action: / Action Input: / Observation:.
   When done, output Final Answer: <answer>"

## Prompt Templates

Different LLMs need different formats:
  GPT-4:    JSON tool calls natively
  LLaMA:    Text-based ReAct prompting
  Claude:   XML tool use natively
  Mistral:  JSON tool calls natively

NanoMind implements text-based ReAct as the universal fallback.

Reference:
  Yao et al. (2022) "ReAct: Synergizing Reasoning and Acting in Language Models"
  https://arxiv.org/abs/2210.03629
"""

from __future__ import annotations
from dataclasses import dataclass, field
from nanomind.agents.tool import ToolRegistry
from nanomind.agents.parser import ToolCallParser, ToolCall, ToolResult
from nanomind.utils.logger import get_logger

log = get_logger("agents.react")


@dataclass
class AgentStep:
    """A single step in an agent trajectory."""
    step_num:    int
    thought:     str
    action:      str | None
    action_input: dict | None
    observation: str | None
    is_final:    bool = False
    final_answer: str | None = None

    def to_text(self) -> str:
        lines = [f"Thought: {self.thought}"]
        if self.action:
            import json
            lines.append(f"Action: {self.action}")
            lines.append(f"Action Input: {json.dumps(self.action_input or {})}")
        if self.observation:
            lines.append(f"Observation: {self.observation}")
        if self.final_answer:
            lines.append(f"Final Answer: {self.final_answer}")
        return "\n".join(lines)


@dataclass
class AgentTrajectory:
    """Full agent reasoning trajectory."""
    query:   str
    steps:   list[AgentStep] = field(default_factory=list)
    answer:  str | None      = None
    success: bool            = False

    def to_text(self) -> str:
        parts = [f"Question: {self.query}"]
        for step in self.steps:
            parts.append(step.to_text())
        if self.answer:
            parts.append(f"Final Answer: {self.answer}")
        return "\n\n".join(parts)

    def n_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.action and not s.is_final)


class ReActAgent:
    """
    ReAct agent: interleaves Thought/Action/Observation traces.

    Simulates LLM-generated reasoning and tool calls using
    a pluggable LM function.

    Args:
        registry:   :class:`ToolRegistry` with available tools.
        lm_fn:      Callable (prompt: str) → str for LLM generation.
        max_steps:  Maximum reasoning steps before giving up.
        verbose:    Print trajectory to stdout.

    Example::

        def lm(prompt):
            return "Thought: Let me search.\\nAction: search\\nAction Input: {...}"

        agent = ReActAgent(registry, lm_fn=lm, max_steps=5)
        traj  = agent.run("What is the capital of France?")
        print(traj.answer)
    """

    SYSTEM_PROMPT = """You are an AI assistant that solves tasks step by step.
You have access to the following tools:
{tools}

Use this format:
Thought: think about what to do
Action: <tool_name>
Action Input: <JSON arguments>
Observation: <tool result>
... (repeat as needed)
Final Answer: <your final answer>

Begin!"""

    def __init__(
        self,
        registry:  ToolRegistry,
        lm_fn:     object = None,
        max_steps: int    = 8,
        verbose:   bool   = False,
    ) -> None:
        self.registry  = registry
        self.lm_fn     = lm_fn
        self.max_steps = max_steps
        self.verbose   = verbose
        self.parser    = ToolCallParser()

    def _build_prompt(self, query: str, history: str) -> str:
        tools_txt = "\n".join(
            f"  - {t.name}: {t.description}"
            for t in self.registry._tools.values()
        )
        system = self.SYSTEM_PROMPT.format(tools=tools_txt)
        return f"{system}\n\nQuestion: {query}\n{history}"

    def _execute_tool(self, call: ToolCall) -> str:
        """Execute a tool call and return the result string."""
        try:
            result = self.registry.call(call.name, **call.arguments)
            return str(result)
        except KeyError:
            return f"Error: Tool '{call.name}' not found."
        except Exception as e:
            return f"Error: {e}"

    def run(self, query: str) -> AgentTrajectory:
        """
        Run the ReAct agent loop.

        Args:
            query: User question or task.

        Returns:
            :class:`AgentTrajectory` with all steps and final answer.
        """
        traj    = AgentTrajectory(query=query)
        history = ""

        for step_num in range(self.max_steps):
            prompt   = self._build_prompt(query, history)
            response = self.lm_fn(prompt) if self.lm_fn else self._mock_lm(query, step_num)

            if self.verbose:
                print(f"\n[Step {step_num + 1}]\n{response}")

            # Check for final answer
            final = self.parser.extract_final_answer(response)
            if final:
                step = AgentStep(
                    step_num=step_num, thought="",
                    action=None, action_input=None,
                    observation=None, is_final=True, final_answer=final,
                )
                traj.steps.append(step)
                traj.answer  = final
                traj.success = True
                break

            # Parse tool calls
            calls = self.parser.parse(response)
            if calls:
                call = calls[0]   # Execute first call (parallel in multi-call variant)
                obs  = self._execute_tool(call)
                step = AgentStep(
                    step_num     = step_num,
                    thought      = f"Calling {call.name}",
                    action       = call.name,
                    action_input = call.arguments,
                    observation  = obs,
                )
                traj.steps.append(step)
                history += f"\n{step.to_text()}"
                log.info(f"Step {step_num}: {call.name}({call.arguments}) → {obs[:80]}")
            else:
                # No tool call and no final answer — terminate
                traj.answer  = response.strip()
                traj.success = True
                break

        return traj

    def _mock_lm(self, query: str, step: int) -> str:
        """Mock LM for testing (returns Final Answer directly)."""
        if step == 0:
            return f'Thought: I can answer this.\nFinal Answer: Mock answer to: {query}'
        return f"Final Answer: Done."
''')
commit("feat: add ReActAgent — Thought/Action/Observation loop, AgentStep, AgentTrajectory, tool execution")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — Parallel tool execution
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/agents/parallel.py", '''\
"""
nanomind/agents/parallel.py — Parallel tool call execution.

GPT-4-turbo and Claude 3 can generate multiple tool calls in one step,
which are then executed in parallel for efficiency.

Example: "Find the weather in London and Paris"
  → LLM generates 2 tool calls simultaneously
  → Both execute in parallel (threads or async)
  → Both results returned to LLM in one message

This module implements parallel execution using Python threads.
Async version uses asyncio for I/O-bound tools.
"""

from __future__ import annotations
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from nanomind.agents.tool import ToolRegistry
from nanomind.agents.parser import ToolCall, ToolResult
from nanomind.utils.logger import get_logger

log = get_logger("agents.parallel")


@dataclass
class ParallelExecutionResult:
    """Results from parallel tool execution."""
    results:      list[ToolResult]
    total_time_s: float
    n_succeeded:  int
    n_failed:     int

    def to_messages(self) -> list[dict]:
        """Convert all results to LLM message dicts."""
        return [r.to_message() for r in self.results]

    def summary(self) -> str:
        return (f"{self.n_succeeded}/{len(self.results)} succeeded "
                f"in {self.total_time_s:.3f}s")


class ParallelToolExecutor:
    """
    Execute multiple tool calls in parallel using thread pool.

    Args:
        registry:    :class:`ToolRegistry`.
        max_workers: Thread pool size.
        timeout:     Per-tool timeout in seconds.

    Example::

        executor = ParallelToolExecutor(registry, max_workers=4)
        calls    = [ToolCall(...), ToolCall(...)]
        result   = executor.execute(calls)
        messages = result.to_messages()
    """

    def __init__(
        self,
        registry:    ToolRegistry,
        max_workers: int   = 4,
        timeout:     float = 30.0,
    ) -> None:
        self.registry    = registry
        self.max_workers = max_workers
        self.timeout     = timeout

    def _execute_one(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        try:
            tool = self.registry.get(call.name)
            if tool is None:
                return ToolResult(call.id, call.name, "",
                                  error=f"Tool '{call.name}' not found", success=False)
            ok, msg = tool.validate_args(call.arguments)
            if not ok:
                return ToolResult(call.id, call.name, "",
                                  error=f"Validation: {msg}", success=False)
            result = tool(**call.arguments)
            return ToolResult(call.id, call.name, str(result), success=True)
        except Exception as e:
            return ToolResult(call.id, call.name, "", error=str(e), success=False)

    def execute(self, calls: list[ToolCall]) -> ParallelExecutionResult:
        """
        Execute multiple tool calls in parallel.

        Args:
            calls: List of :class:`ToolCall` objects.

        Returns:
            :class:`ParallelExecutionResult` with all results.
        """
        t0      = time.monotonic()
        results = []

        if len(calls) == 1:
            # No overhead for single call
            results = [self._execute_one(calls[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(calls))) as pool:
                futures = {pool.submit(self._execute_one, c): c for c in calls}
                for fut in as_completed(futures, timeout=self.timeout):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        call = futures[fut]
                        results.append(ToolResult(call.id, call.name, "",
                                                   error=str(e), success=False))

        elapsed   = time.monotonic() - t0
        succeeded = sum(1 for r in results if r.success)
        return ParallelExecutionResult(
            results      = results,
            total_time_s = elapsed,
            n_succeeded  = succeeded,
            n_failed     = len(results) - succeeded,
        )

    def execute_sequential(self, calls: list[ToolCall]) -> ParallelExecutionResult:
        """Execute calls one by one (for debugging or ordered dependencies)."""
        t0      = time.monotonic()
        results = [self._execute_one(c) for c in calls]
        elapsed = time.monotonic() - t0
        return ParallelExecutionResult(
            results      = results,
            total_time_s = elapsed,
            n_succeeded  = sum(1 for r in results if r.success),
            n_failed     = sum(1 for r in results if not r.success),
        )
''')
commit("feat: add ParallelToolExecutor — ThreadPoolExecutor parallel calls, timeout, ParallelExecutionResult")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Built-in demo tools
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/agents/builtin_tools.py", '''\
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
    return "\n".join(results)


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
''')
commit("feat: add built-in tools — calculator, word_count, reverse, datetime, mock_search, memory, default_registry")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Structured output / JSON mode
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/agents/structured.py", '''\
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
        base = "\n\nRespond with valid JSON only."
        if self.schema:
            ex = json.dumps(self.schema.example(), indent=2)
            base += f"\nExpected format:\n```json\n{ex}\n```"
        return base
''')
commit("feat: add OutputSchema, StructuredExtractor — JSON parsing, schema validation, prompt_suffix")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — Multi-step planner
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/agents/planner.py", '''\
"""
nanomind/agents/planner.py — Multi-step task planner for agentic workflows.

## Plan-and-Execute (Wang et al., 2023)

Decompose a complex task into a plan, then execute each step:

  Plan:
    1. Search for recent AI papers
    2. Extract key findings from each
    3. Summarise into a report

  Execute:
    Step 1 → search tool → papers found
    Step 2 → extract tool × N → findings
    Step 3 → summarise → final report

Benefits over ReAct:
  - Long-horizon tasks (100+ steps)
  - Explicit task decomposition (inspectable)
  - Can re-plan on failure

Used in: AutoGPT, BabyAGI, LangGraph, MetaGPT.

Reference:
  Wang et al. (2023) "Plan-and-Solve Prompting"
  https://arxiv.org/abs/2305.04091
"""

from __future__ import annotations
from dataclasses import dataclass, field
from nanomind.agents.tool import ToolRegistry
from nanomind.agents.parser import ToolCallParser, ToolCall


@dataclass
class PlanStep:
    """A single step in an agent plan."""
    step_id:     int
    description: str
    tool:        str | None = None
    args:        dict       = field(default_factory=dict)
    depends_on:  list[int]  = field(default_factory=list)   # step IDs
    result:      str | None = None
    status:      str        = "pending"   # pending | running | done | failed

    def is_ready(self, completed: set[int]) -> bool:
        """Check if all dependencies are completed."""
        return all(d in completed for d in self.depends_on)


@dataclass
class Plan:
    """A structured execution plan."""
    goal:  str
    steps: list[PlanStep] = field(default_factory=list)

    def add_step(
        self,
        description: str,
        tool:        str | None = None,
        args:        dict | None = None,
        depends_on:  list[int] | None = None,
    ) -> PlanStep:
        step = PlanStep(
            step_id     = len(self.steps),
            description = description,
            tool        = tool,
            args        = args or {},
            depends_on  = depends_on or [],
        )
        self.steps.append(step)
        return step

    def to_text(self) -> str:
        lines = [f"Goal: {self.goal}", "Plan:"]
        for s in self.steps:
            dep = f" (after {s.depends_on})" if s.depends_on else ""
            tool_info = f" → {s.tool}({s.args})" if s.tool else ""
            lines.append(f"  {s.step_id}. {s.description}{tool_info}{dep}")
        return "\n".join(lines)

    def n_done(self) -> int:
        return sum(1 for s in self.steps if s.status == "done")

    def progress(self) -> str:
        return f"{self.n_done()}/{len(self.steps)} steps done"


class SequentialPlanner:
    """
    Execute plan steps one by one in order.

    Args:
        registry: :class:`ToolRegistry`.

    Example::

        planner = SequentialPlanner(registry)
        plan    = Plan("Compute and store 2+2")
        plan.add_step("Calculate 2+2", tool="calculator", args={"expression": "2+2"})
        plan.add_step("Store result", tool="memory_set",
                       args={"key": "result", "value": "PREV_RESULT"},
                       depends_on=[0])
        result  = planner.execute(plan)
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, plan: Plan) -> Plan:
        """Execute all plan steps in order."""
        completed = set()
        results   = {}

        for step in plan.steps:
            if not step.is_ready(completed):
                step.status = "failed"
                step.result = "Dependency not met"
                continue

            step.status = "running"
            if step.tool:
                try:
                    # Replace PREV_RESULT placeholder with last result
                    args = {k: (results.get(max(completed, default=0), "")
                                if v == "PREV_RESULT" else v)
                            for k, v in step.args.items()}
                    step.result = str(self.registry.call(step.tool, **args))
                    step.status = "done"
                except Exception as e:
                    step.result = f"Error: {e}"
                    step.status = "failed"
            else:
                step.result = "Step completed (no tool)"
                step.status = "done"

            results[step.step_id] = step.result
            if step.status == "done":
                completed.add(step.step_id)

        return plan


class DAGPlanner(SequentialPlanner):
    """
    Execute plan steps in dependency order (DAG-aware).

    Steps without dependencies run first.
    Steps with all dependencies satisfied run next.
    Supports parallel execution of independent steps.
    """

    def execute(self, plan: Plan) -> Plan:
        """Execute steps respecting dependency graph."""
        completed = set()
        pending   = list(range(len(plan.steps)))

        while pending:
            # Find all ready steps
            ready = [i for i in pending if plan.steps[i].is_ready(completed)]
            if not ready:
                # Deadlock: fail remaining steps
                for i in pending:
                    plan.steps[i].status = "failed"
                    plan.steps[i].result = "Dependency deadlock"
                break

            # Execute all ready steps (sequential for simplicity)
            for i in ready:
                step = plan.steps[i]
                step.status = "running"
                if step.tool:
                    try:
                        step.result = str(self.registry.call(step.tool, **step.args))
                        step.status = "done"
                    except Exception as e:
                        step.result = f"Error: {e}"
                        step.status = "failed"
                else:
                    step.result = "Done"
                    step.status = "done"
                if step.status == "done":
                    completed.add(i)
                pending.remove(i)

        return plan
''')
commit("feat: add Plan, PlanStep, SequentialPlanner, DAGPlanner — dependency-aware multi-step execution")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — agents __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/agents/__init__.py", '''\
"""NanoMind Agents sub-package — Tool Use & Agentic LLM infrastructure.

Implements the full agentic stack:
  1. Tool / ToolParameter  — JSON Schema tool definition
  2. ToolRegistry          — tool registration, call, schemas
  3. @tool decorator       — decorator for quick tool creation
  4. ToolCall / ToolResult — structured call/result types
  5. ToolCallParser        — JSON/XML/ReAct format parsing
  6. ReActAgent            — Thought/Action/Observation loop
  7. AgentStep / AgentTrajectory — step tracking
  8. ParallelToolExecutor  — concurrent tool execution
  9. OutputSchema          — JSON field validation
  10. StructuredExtractor  — JSON output parsing from LLM text
  11. Plan / PlanStep      — structured task decomposition
  12. SequentialPlanner    — step-by-step execution
  13. DAGPlanner           — dependency-aware DAG execution
  14. Built-in tools       — calculator, search, memory, datetime, etc.

Primary exports:
    - :class:`Tool`                  — tool with JSON schema + callable
    - :class:`ToolParameter`         — parameter specification
    - :class:`ToolRegistry`          — register, call, schemas
    - :func:`tool`                   — @tool decorator
    - :class:`ToolCall`              — parsed tool call (id, name, args)
    - :class:`ToolResult`            — execution result (success/error)
    - :class:`ToolCallParser`        — parse LLM output → ToolCall
    - :class:`ReActAgent`            — ReAct reasoning agent loop
    - :class:`AgentStep`             — single Thought/Action/Obs step
    - :class:`AgentTrajectory`       — full agent run record
    - :class:`ParallelToolExecutor`  — thread-parallel tool execution
    - :class:`OutputSchema`          — expected JSON field schema
    - :class:`StructuredExtractor`   — JSON extraction + validation
    - :class:`Plan`                  — multi-step task plan
    - :class:`PlanStep`              — single plan step with deps
    - :class:`SequentialPlanner`     — ordered plan execution
    - :class:`DAGPlanner`            — DAG-aware plan execution
    - :func:`default_registry`       — registry with all built-in tools
"""

from nanomind.agents.tool import Tool, ToolParameter, ToolRegistry, tool
from nanomind.agents.parser import ToolCall, ToolResult, ToolCallParser
from nanomind.agents.react import ReActAgent, AgentStep, AgentTrajectory
from nanomind.agents.parallel import ParallelToolExecutor, ParallelExecutionResult
from nanomind.agents.structured import OutputSchema, StructuredExtractor, StructuredOutput
from nanomind.agents.planner import Plan, PlanStep, SequentialPlanner, DAGPlanner
from nanomind.agents.builtin_tools import (
    default_registry, CALCULATOR_TOOL, SEARCH_TOOL, DATETIME_TOOL,
    WORD_COUNT_TOOL, MEMORY_GET_TOOL, MEMORY_SET_TOOL,
)

__all__ = [
    "Tool", "ToolParameter", "ToolRegistry", "tool",
    "ToolCall", "ToolResult", "ToolCallParser",
    "ReActAgent", "AgentStep", "AgentTrajectory",
    "ParallelToolExecutor", "ParallelExecutionResult",
    "OutputSchema", "StructuredExtractor", "StructuredOutput",
    "Plan", "PlanStep", "SequentialPlanner", "DAGPlanner",
    "default_registry", "CALCULATOR_TOOL", "SEARCH_TOOL",
    "DATETIME_TOOL", "WORD_COUNT_TOOL", "MEMORY_GET_TOOL", "MEMORY_SET_TOOL",
]
''')
commit("refactor: export all agents components from nanomind/agents/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — example
# ══════════════════════════════════════════════════════════════════════════════
write("examples/agents_demo.py", '''\
"""
examples/agents_demo.py — NanoMind Tool Use & Agentic LLM demo.

Demonstrates:
  1. Tool definition and registry
  2. Tool schema generation (OpenAI format)
  3. Tool call parsing (JSON/XML/ReAct)
  4. ReAct agent loop
  5. Parallel tool execution
  6. Structured output extraction
  7. Multi-step planning (DAG)
  8. Built-in tools (calculator, search, memory)

Usage:
    python examples/agents_demo.py
"""
import json
from nanomind.agents import (
    Tool, ToolParameter, ToolRegistry, tool,
    ToolCall, ToolResult, ToolCallParser,
    ReActAgent, AgentTrajectory,
    ParallelToolExecutor,
    OutputSchema, StructuredExtractor,
    Plan, SequentialPlanner, DAGPlanner,
    default_registry, CALCULATOR_TOOL,
)

print("=" * 60)
print("NanoMind Tool Use & Agentic LLM Demo")
print("=" * 60)

# ── Tool Definition ───────────────────────────────────────────────────────────
print("\n── Tool Definition & Registry ──")
registry = default_registry()
print(f"  Registered tools: {registry.names()}")
print(f"  Total tools: {len(registry)}")
schema = CALCULATOR_TOOL.to_schema()
print(f"  Calculator schema: {json.dumps(schema, indent=2)[:200]}...")

# ── @tool decorator ───────────────────────────────────────────────────────────
print("\n── @tool Decorator ──")
@tool(description="Square a number")
def square(x: float) -> float:
    return x * x

print(f"  square(5) = {square(x=5.0)}")
print(f"  Schema: {json.dumps(square.to_schema()['function']['parameters'])}")

# ── Tool Registry Calls ───────────────────────────────────────────────────────
print("\n── Tool Calls via Registry ──")
calc_result = registry.call("calculator", expression="2 ** 10")
print(f"  calculator(2**10) = {calc_result}")
wc_result = registry.call("word_count", text="Hello world, this is NanoMind!")
print(f"  word_count = {wc_result}")
dt = registry.call("get_datetime")
print(f"  get_datetime = {dt}")

# ── Tool Call Parsing ─────────────────────────────────────────────────────────
print("\n── Tool Call Parsing ──")
parser = ToolCallParser()

# JSON format
json_text = '{"name": "calculator", "arguments": {"expression": "42 * 7"}}'
calls = parser.parse(json_text)
print(f"  JSON parse: {calls[0].name}({calls[0].arguments})")

# Code block format
block_text = '```json\n{"name": "web_search", "arguments": {"query": "LLM news"}}\n```'
calls = parser.parse(block_text)
print(f"  Code block: {calls[0].name}({calls[0].arguments})")

# ReAct format
react_text = 'Action: calculator\nAction Input: {"expression": "100 / 4"}'
calls = parser.parse(react_text)
print(f"  ReAct: {calls[0].name}({calls[0].arguments})")

# Final answer
fa_text = "Final Answer: The result is 42."
print(f"  Final Answer: {parser.extract_final_answer(fa_text)}")

# ── ReAct Agent ───────────────────────────────────────────────────────────────
print("\n── ReAct Agent ──")
def mock_lm(prompt: str) -> str:
    """Simulate LLM that uses calculator then gives final answer."""
    if "Observation:" not in prompt:
        return '{"name": "calculator", "arguments": {"expression": "15 * 8"}}'
    return "Final Answer: 15 times 8 equals 120."

agent = ReActAgent(registry, lm_fn=mock_lm, max_steps=3, verbose=False)
traj  = agent.run("What is 15 times 8?")
print(f"  Answer: {traj.answer}")
print(f"  Steps:  {len(traj.steps)}, Tool calls: {traj.n_tool_calls()}")

# ── Parallel Tool Execution ────────────────────────────────────────────────────
print("\n── Parallel Tool Execution ──")
executor = ParallelToolExecutor(registry, max_workers=4)
calls    = [
    ToolCall("c1", "calculator", {"expression": "100 + 200"}),
    ToolCall("c2", "calculator", {"expression": "50 * 3"}),
    ToolCall("c3", "word_count", {"text": "NanoMind is awesome!"}),
]
par_result = executor.execute(calls)
print(f"  {par_result.summary()}")
for r in par_result.results:
    print(f"  {r.name}: {r.result}")

# ── Structured Output ──────────────────────────────────────────────────────────
print("\n── Structured Output Extraction ──")
schema   = OutputSchema({"name": (str, True), "score": (float, True), "tags": (list, False)})
extractor = StructuredExtractor(schema)
llm_out  = 'Here is my analysis:\n```json\n{"name": "NanoMind", "score": 0.98, "tags": ["AI", "LLM"]}\n```'
result   = extractor.extract(llm_out)
print(f"  Valid: {result.valid}")
print(f"  Name: {result.get('name')}, Score: {result.get('score')}")
print(f"  Tags: {result.get('tags')}")

# ── Multi-step Planning ───────────────────────────────────────────────────────
print("\n── Multi-step Planning (DAG) ──")
plan = Plan(goal="Calculate and remember 7 factorial")
plan.add_step("Compute 7!", tool="calculator",  args={"expression": "7*6*5*4*3*2*1"})
plan.add_step("Store result", tool="memory_set",
               args={"key": "seven_factorial", "value": "5040"}, depends_on=[0])
plan.add_step("Retrieve to verify", tool="memory_get",
               args={"key": "seven_factorial"}, depends_on=[1])

print(f"\n{plan.to_text()}")
dag = DAGPlanner(registry)
dag.execute(plan)
print(f"\n  Progress: {plan.progress()}")
for s in plan.steps:
    print(f"  Step {s.step_id} [{s.status}]: {s.result}")

print("\nAgents demo complete!")
''')
commit("feat: add examples/agents_demo.py — tool def, registry, parsing, ReAct, parallel, structured, plan")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_agents.py", '''\
"""tests/test_agents.py — Tests for NanoMind Agents package."""
import json
import pytest
from nanomind.agents import (
    Tool, ToolParameter, ToolRegistry, tool,
    ToolCall, ToolResult, ToolCallParser,
    ReActAgent, AgentStep, AgentTrajectory,
    ParallelToolExecutor,
    OutputSchema, StructuredExtractor, StructuredOutput,
    Plan, PlanStep, SequentialPlanner, DAGPlanner,
    default_registry, CALCULATOR_TOOL,
)


# ── Tool ──────────────────────────────────────────────────────────────────────

class TestTool:
    def test_schema_has_required_keys(self):
        s = CALCULATOR_TOOL.to_schema()
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "parameters" in s["function"]

    def test_call_executes_func(self):
        result = CALCULATOR_TOOL(expression="2+2")
        assert "4" in result

    def test_validate_args_missing_required(self):
        ok, msg = CALCULATOR_TOOL.validate_args({})
        assert not ok
        assert "expression" in msg

    def test_validate_args_enum(self):
        p   = ToolParameter("color", "string", "Color", enum=["red", "blue"])
        t   = Tool("t", "test", [p], func=lambda color: color)
        ok, msg = t.validate_args({"color": "green"})
        assert not ok

    def test_tool_decorator(self):
        @tool(description="Double a number")
        def double(x: float) -> float:
            return x * 2
        assert isinstance(double, Tool)
        assert double.name == "double"
        result = double(x=5.0)
        assert result == 10.0

    def test_tool_decorator_schema(self):
        @tool(description="greet")
        def greet(name: str) -> str:
            return f"Hello {name}"
        s = greet.to_schema()
        assert "name" in s["function"]["parameters"]["properties"]


# ── ToolRegistry ──────────────────────────────────────────────────────────────

class TestToolRegistry:
    def test_register_and_call(self):
        reg = ToolRegistry([CALCULATOR_TOOL])
        res = reg.call("calculator", expression="3*3")
        assert "9" in res

    def test_contains(self):
        reg = ToolRegistry([CALCULATOR_TOOL])
        assert "calculator" in reg
        assert "bogus" not in reg

    def test_len(self):
        reg = default_registry()
        assert len(reg) > 0

    def test_schemas_list(self):
        reg = ToolRegistry([CALCULATOR_TOOL])
        schemas = reg.schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 1

    def test_call_unknown_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.call("nonexistent")

    def test_by_category(self):
        reg = default_registry()
        math = reg.by_category("math")
        assert any(t.name == "calculator" for t in math)

    def test_names(self):
        reg = default_registry()
        assert "calculator" in reg.names()


# ── ToolCallParser ────────────────────────────────────────────────────────────

class TestToolCallParser:
    def _p(self):
        return ToolCallParser()

    def test_parse_json(self):
        p     = self._p()
        text  = '{"name": "calculator", "arguments": {"expression": "1+1"}}'
        calls = p.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "calculator"
        assert calls[0].arguments["expression"] == "1+1"

    def test_parse_code_block(self):
        p    = self._p()
        text = '```json\n{"name": "web_search", "arguments": {"query": "AI"}}\n```'
        calls = p.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "web_search"

    def test_parse_react(self):
        p    = self._p()
        text = 'Action: calculator\nAction Input: {"expression": "5*5"}'
        calls = p.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "calculator"

    def test_extract_final_answer(self):
        p   = self._p()
        txt = "Final Answer: 42"
        assert p.extract_final_answer(txt) == "42"

    def test_has_tool_call_true(self):
        p = self._p()
        assert p.has_tool_call('{"name": "x", "arguments": {}}')

    def test_has_tool_call_false(self):
        p = self._p()
        assert not p.has_tool_call("Just a plain sentence.")


# ── ReActAgent ────────────────────────────────────────────────────────────────

class TestReActAgent:
    def _agent(self, lm_fn=None):
        return ReActAgent(default_registry(), lm_fn=lm_fn, max_steps=3)

    def test_mock_lm_returns_answer(self):
        agent = self._agent()
        traj  = agent.run("What is 1+1?")
        assert traj.answer is not None

    def test_lm_with_tool_call(self):
        step = [0]
        def lm(prompt):
            if step[0] == 0:
                step[0] += 1
                return '{"name": "calculator", "arguments": {"expression": "10+5"}}'
            return "Final Answer: 15"

        agent = self._agent(lm_fn=lm)
        traj  = agent.run("Add 10 and 5")
        assert traj.answer == "15"

    def test_trajectory_has_steps(self):
        agent = self._agent()
        traj  = agent.run("test")
        assert isinstance(traj.steps, list)

    def test_success_flag_set(self):
        agent = self._agent()
        traj  = agent.run("test")
        assert traj.success is True


# ── ParallelToolExecutor ──────────────────────────────────────────────────────

class TestParallelExecutor:
    def test_single_call(self):
        executor = ParallelToolExecutor(default_registry())
        calls    = [ToolCall("c1", "calculator", {"expression": "2+2"})]
        res      = executor.execute(calls)
        assert res.n_succeeded == 1

    def test_multiple_calls(self):
        executor = ParallelToolExecutor(default_registry(), max_workers=2)
        calls    = [
            ToolCall("c1", "calculator", {"expression": "1+1"}),
            ToolCall("c2", "calculator", {"expression": "2+2"}),
        ]
        res = executor.execute(calls)
        assert res.n_succeeded == 2
        assert len(res.results) == 2

    def test_unknown_tool_fails_gracefully(self):
        executor = ParallelToolExecutor(default_registry())
        calls    = [ToolCall("c1", "nonexistent", {})]
        res      = executor.execute(calls)
        assert res.n_failed == 1

    def test_to_messages(self):
        executor = ParallelToolExecutor(default_registry())
        calls    = [ToolCall("c1", "calculator", {"expression": "3*3"})]
        res      = executor.execute(calls)
        msgs     = res.to_messages()
        assert msgs[0]["role"] == "tool"


# ── Structured Output ─────────────────────────────────────────────────────────

class TestStructuredOutput:
    def test_extract_json_code_block(self):
        s    = OutputSchema({"score": (float, True)})
        ext  = StructuredExtractor(s)
        text = '```json\n{"score": 0.95}\n```'
        res  = ext.extract(text)
        assert res.valid
        assert res.get("score") == 0.95

    def test_extract_bare_json(self):
        ext  = StructuredExtractor()
        res  = ext.extract('{"key": "value"}')
        assert res.valid

    def test_missing_required_field_invalid(self):
        s   = OutputSchema({"name": (str, True)})
        ext = StructuredExtractor(s)
        res = ext.extract('{"other": "x"}')
        assert not res.valid

    def test_schema_example(self):
        s = OutputSchema({"name": (str, True), "n": (int, False)})
        ex = s.example()
        assert "name" in ex

    def test_no_json_found(self):
        ext = StructuredExtractor()
        res = ext.extract("Just plain text with no JSON.")
        assert not res.valid


# ── Planner ───────────────────────────────────────────────────────────────────

class TestPlanner:
    def _reg(self):
        return default_registry()

    def test_sequential_plan(self):
        reg  = self._reg()
        plan = Plan("test")
        plan.add_step("Compute", tool="calculator", args={"expression": "2+2"})
        SequentialPlanner(reg).execute(plan)
        assert plan.steps[0].status == "done"
        assert "4" in plan.steps[0].result

    def test_dag_respects_deps(self):
        reg  = self._reg()
        plan = Plan("test")
        plan.add_step("step 0", tool="calculator", args={"expression": "1+1"})
        plan.add_step("step 1", tool="calculator",
                       args={"expression": "2*2"}, depends_on=[0])
        DAGPlanner(reg).execute(plan)
        assert plan.steps[1].status == "done"

    def test_plan_progress(self):
        reg  = self._reg()
        plan = Plan("test")
        plan.add_step("A", tool="calculator", args={"expression": "1"})
        plan.add_step("B", tool="calculator", args={"expression": "2"})
        SequentialPlanner(reg).execute(plan)
        assert plan.n_done() == 2

    def test_plan_to_text(self):
        plan = Plan("My goal")
        plan.add_step("Do something")
        txt  = plan.to_text()
        assert "My goal" in txt
        assert "Do something" in txt
''')
commit("test: add full agents test suite — tool, registry, parser, ReAct, parallel, structured, planner")

# COMMITS 12-18
for title, body in [
    ("test: add ToolRegistry unregister test", '''
class TestToolUnregister:
    def test_unregister_removes_tool(self):
        from nanomind.agents import ToolRegistry, CALCULATOR_TOOL
        reg = ToolRegistry([CALCULATOR_TOOL])
        assert "calculator" in reg
        reg.unregister("calculator")
        assert "calculator" not in reg

    def test_unregister_nonexistent_no_error(self):
        reg = ToolRegistry()
        reg.unregister("ghost")   # should not raise
'''),
    ("test: add Tool validate_args enum correct value test", '''
class TestToolEnumValidation:
    def test_valid_enum(self):
        from nanomind.agents import Tool, ToolParameter
        p = ToolParameter("unit", "string", "unit", enum=["c", "f"])
        t = Tool("t", "test", [p], func=lambda unit: unit)
        ok, _ = t.validate_args({"unit": "c"})
        assert ok

    def test_invalid_enum_fails(self):
        from nanomind.agents import Tool, ToolParameter
        p = ToolParameter("unit", "string", "unit", enum=["c", "f"])
        t = Tool("t", "test", [p], func=lambda unit: unit)
        ok, msg = t.validate_args({"unit": "k"})
        assert not ok
'''),
    ("test: add ToolCallParser XML format test", '''
class TestXMLParsing:
    def test_xml_tool_call(self):
        from nanomind.agents import ToolCallParser
        p    = ToolCallParser()
        text = "<tool_call><name>calculator</name><arguments>{\\"expression\\": \\"3+3\\"}</arguments></tool_call>"
        calls = p.parse(text)
        assert len(calls) >= 1
        assert calls[0].name == "calculator"
'''),
    ("test: add ToolResult to_message format test", '''
class TestToolResult:
    def test_success_message_format(self):
        from nanomind.agents import ToolResult
        r = ToolResult("call_1", "calculator", "4.0", success=True)
        m = r.to_message()
        assert m["role"] == "tool"
        assert m["content"] == "4.0"

    def test_error_message_format(self):
        from nanomind.agents import ToolResult
        r = ToolResult("call_1", "calc", "", error="division by zero", success=False)
        m = r.to_message()
        assert "division by zero" in m["content"]
'''),
    ("test: add ParallelExecutor sequential matches parallel test", '''
class TestSequentialVsParallel:
    def test_sequential_same_results(self):
        from nanomind.agents import ParallelToolExecutor, ToolCall, default_registry
        executor = ParallelToolExecutor(default_registry())
        calls    = [ToolCall("c1", "calculator", {"expression": "5+5"}),
                    ToolCall("c2", "calculator", {"expression": "10*2"})]
        par = executor.execute(calls)
        seq = executor.execute_sequential(calls)
        assert par.n_succeeded == seq.n_succeeded
'''),
    ("test: add DAGPlanner deadlock detection test", '''
class TestDAGDeadlock:
    def test_circular_dep_fails_gracefully(self):
        from nanomind.agents import Plan, DAGPlanner, default_registry
        plan = Plan("test")
        plan.add_step("A", depends_on=[1])   # A depends on B
        plan.add_step("B", depends_on=[0])   # B depends on A
        dag  = DAGPlanner(default_registry())
        dag.execute(plan)                     # should not raise
        # Both should be failed due to deadlock
        statuses = {s.status for s in plan.steps}
        assert "failed" in statuses
'''),
    ("test: add StructuredExtractor prompt suffix test", '''
class TestPromptSuffix:
    def test_suffix_contains_json(self):
        from nanomind.agents import OutputSchema, StructuredExtractor
        s   = OutputSchema({"x": (str, True)})
        ext = StructuredExtractor(s)
        suf = ext.build_prompt_suffix()
        assert "JSON" in suf or "json" in suf

    def test_suffix_no_schema(self):
        from nanomind.agents import StructuredExtractor
        ext = StructuredExtractor()
        suf = ext.build_prompt_suffix()
        assert len(suf) > 0
'''),
]:
    src = read("tests/test_agents.py")
    src += "\n" + body
    write("tests/test_agents.py", src)
    commit(title)

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v4.7.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"4.6.0\"", "__version__ = \"4.7.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v4.7.0 — Tool Use & Agentic LLMs release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `quant`      | Quantization — INT8/4/2, RTN, GPTQ, AWQ, QAT/STE, calibration, mixed-precision |",
    "| `quant`      | Quantization — INT8/4/2, RTN, GPTQ, AWQ, QAT/STE, calibration, mixed-precision |\n"
    "| `agents`     | Tool Use & Agents — ReAct, function calling, parallel exec, structured output, DAG planner |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [4.7.0] — 2024 — Tool Use & Agentic LLMs\n\n### Added\n"
      "- `Tool` / `ToolParameter` — JSON Schema tool definition\n"
      "- `ToolRegistry` — register, call, schemas, by_category\n"
      "- `@tool` decorator — quick function-to-Tool conversion\n"
      "- `ToolCall` / `ToolResult` — structured call/result types\n"
      "- `ToolCallParser` — JSON/XML/ReAct format parsing\n"
      "- `ReActAgent` — Thought/Action/Observation reasoning loop\n"
      "- `AgentStep` / `AgentTrajectory` — step tracking and trajectory\n"
      "- `ParallelToolExecutor` — thread-parallel concurrent tool calls\n"
      "- `OutputSchema` / `StructuredExtractor` — JSON output validation\n"
      "- `Plan` / `PlanStep` — multi-step task decomposition\n"
      "- `SequentialPlanner` / `DAGPlanner` — dependency-aware execution\n"
      "- Built-in tools: calculator, word_count, reverse, datetime, mock_search, memory\n"
      "- `examples/agents_demo.py` — full agentic demo\n\n---\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v4.7.0, update README and CHANGELOG for Day 47 Tool Use & Agents")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 47 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v4.7.0",
    "-m", "NanoMind v4.7.0 — Tool Use & Agentic LLMs", check=False)
r = run("git", "push", "origin", "v4.7.0", check=False)
print("Tag v4.7.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 47 COMPLETE — v4.7.0 TAGGED! ===")
