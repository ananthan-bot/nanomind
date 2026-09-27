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
        return "
".join(lines)

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
