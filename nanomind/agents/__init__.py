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
