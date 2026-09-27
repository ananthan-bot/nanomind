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
