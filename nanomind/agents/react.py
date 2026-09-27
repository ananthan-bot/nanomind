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
        return "
".join(lines)


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
        return "

".join(parts)

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
            return "Thought: Let me search.\nAction: search\nAction Input: {...}"

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
        tools_txt = "
".join(
            f"  - {t.name}: {t.description}"
            for t in self.registry._tools.values()
        )
        system = self.SYSTEM_PROMPT.format(tools=tools_txt)
        return f"{system}

Question: {query}
{history}"

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
                print(f"
[Step {step_num + 1}]
{response}")

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
                history += f"
{step.to_text()}"
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
            return f'Thought: I can answer this.
Final Answer: Mock answer to: {query}'
        return f"Final Answer: Done."
