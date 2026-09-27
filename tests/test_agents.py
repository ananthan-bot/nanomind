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
        text = '```json
{"name": "web_search", "arguments": {"query": "AI"}}
```'
        calls = p.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "web_search"

    def test_parse_react(self):
        p    = self._p()
        text = 'Action: calculator
Action Input: {"expression": "5*5"}'
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
        text = '```json
{"score": 0.95}
```'
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
