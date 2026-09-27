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
print("
── Tool Definition & Registry ──")
registry = default_registry()
print(f"  Registered tools: {registry.names()}")
print(f"  Total tools: {len(registry)}")
schema = CALCULATOR_TOOL.to_schema()
print(f"  Calculator schema: {json.dumps(schema, indent=2)[:200]}...")

# ── @tool decorator ───────────────────────────────────────────────────────────
print("
── @tool Decorator ──")
@tool(description="Square a number")
def square(x: float) -> float:
    return x * x

print(f"  square(5) = {square(x=5.0)}")
print(f"  Schema: {json.dumps(square.to_schema()['function']['parameters'])}")

# ── Tool Registry Calls ───────────────────────────────────────────────────────
print("
── Tool Calls via Registry ──")
calc_result = registry.call("calculator", expression="2 ** 10")
print(f"  calculator(2**10) = {calc_result}")
wc_result = registry.call("word_count", text="Hello world, this is NanoMind!")
print(f"  word_count = {wc_result}")
dt = registry.call("get_datetime")
print(f"  get_datetime = {dt}")

# ── Tool Call Parsing ─────────────────────────────────────────────────────────
print("
── Tool Call Parsing ──")
parser = ToolCallParser()

# JSON format
json_text = '{"name": "calculator", "arguments": {"expression": "42 * 7"}}'
calls = parser.parse(json_text)
print(f"  JSON parse: {calls[0].name}({calls[0].arguments})")

# Code block format
block_text = '```json
{"name": "web_search", "arguments": {"query": "LLM news"}}
```'
calls = parser.parse(block_text)
print(f"  Code block: {calls[0].name}({calls[0].arguments})")

# ReAct format
react_text = 'Action: calculator
Action Input: {"expression": "100 / 4"}'
calls = parser.parse(react_text)
print(f"  ReAct: {calls[0].name}({calls[0].arguments})")

# Final answer
fa_text = "Final Answer: The result is 42."
print(f"  Final Answer: {parser.extract_final_answer(fa_text)}")

# ── ReAct Agent ───────────────────────────────────────────────────────────────
print("
── ReAct Agent ──")
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
print("
── Parallel Tool Execution ──")
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
print("
── Structured Output Extraction ──")
schema   = OutputSchema({"name": (str, True), "score": (float, True), "tags": (list, False)})
extractor = StructuredExtractor(schema)
llm_out  = 'Here is my analysis:
```json
{"name": "NanoMind", "score": 0.98, "tags": ["AI", "LLM"]}
```'
result   = extractor.extract(llm_out)
print(f"  Valid: {result.valid}")
print(f"  Name: {result.get('name')}, Score: {result.get('score')}")
print(f"  Tags: {result.get('tags')}")

# ── Multi-step Planning ───────────────────────────────────────────────────────
print("
── Multi-step Planning (DAG) ──")
plan = Plan(goal="Calculate and remember 7 factorial")
plan.add_step("Compute 7!", tool="calculator",  args={"expression": "7*6*5*4*3*2*1"})
plan.add_step("Store result", tool="memory_set",
               args={"key": "seven_factorial", "value": "5040"}, depends_on=[0])
plan.add_step("Retrieve to verify", tool="memory_get",
               args={"key": "seven_factorial"}, depends_on=[1])

print(f"
{plan.to_text()}")
dag = DAGPlanner(registry)
dag.execute(plan)
print(f"
  Progress: {plan.progress()}")
for s in plan.steps:
    print(f"  Step {s.step_id} [{s.status}]: {s.result}")

print("
Agents demo complete!")
