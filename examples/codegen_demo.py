"""
examples/codegen_demo.py — NanoMind Code Generation & Self-Debugging demo.

Usage:
    python examples/codegen_demo.py
"""
from nanomind.codegen import (
    CodeProblem, TestCase, ProgrammingLanguage, make_sample_problems,
    CodeSandbox, ExecutionResult, check_syntax, check_imports,
    CodePromptBuilder, CodeOutputParser, GenerationConfig,
    ReflexionAgent, ReflexionResult,
    pass_at_k, CodeEvaluator, EvalResult, BenchmarkResult,
    ASTAnalyser, FunctionInfo,
)

print("=" * 60)
print("NanoMind Code Generation & Self-Debugging Demo")
print("=" * 60)

problems = make_sample_problems()

# ── Code Problem ──────────────────────────────────────────────────────────────
print("
── Code Problem ──")
p = problems[0]
print(f"  ID: {p.problem_id}, Difficulty: {p.difficulty}")
print(f"  Description: {p.description}")
print(f"  Prompt:
{p.prompt}
")
print(f"  Tests: {len(p.test_cases)} total, {len(p.hidden_tests)} hidden")
print(f"  Test suite:
{p.test_suite()}")

# ── Syntax & Import Checks ────────────────────────────────────────────────────
print("
── Syntax & Import Checks ──")
good_code = "def add(a, b):
    return a + b"
bad_code  = "def add(a, b)
    return a + b"
imp_code  = "import os
def f(): return os.getcwd()"

print(f"  Good code syntax: {check_syntax(good_code)}")
print(f"  Bad code syntax:  {check_syntax(bad_code)}")
print(f"  Blocked import:   {check_imports(imp_code)}")

# ── Code Sandbox ──────────────────────────────────────────────────────────────
print("
── Code Sandbox Execution ──")
sandbox = CodeSandbox(timeout_s=5.0)

correct = "def add(a, b):
    return a + b"
wrong   = "def add(a, b):
    return a - b"
broken  = "def add(a, b):
    return a +"

for label, code in [("Correct", correct), ("Wrong", wrong), ("Broken", broken)]:
    result = sandbox.execute(code, p.test_cases, p.entry_point)
    print(f"  [{label}] pass={result.pass_rate:.0%} "
          f"({result.n_passed}/{result.n_total}), "
          f"error={str(result.error)[:40] if result.error else None}")

# ── Code Prompt Builder ───────────────────────────────────────────────────────
print("
── Prompt Builder ──")
for style in ["completion", "instruct", "fim", "cot"]:
    builder = CodePromptBuilder(style=style)
    prompt  = builder.build(problems[0])
    print(f"  [{style}] {prompt[:60].replace(chr(10), ' ')}...")

# ── Code Output Parser ────────────────────────────────────────────────────────
print("
── Output Parser ──")
parser = CodeOutputParser()
raw1   = "Here's the solution:
```python
def add(a, b):
    return a + b
```"
raw2   = "def add(a, b):
    return a + b

This is the implementation."
print(f"  From markdown block: {parser.parse(raw1, 'add')[:40]}")
print(f"  From raw code:       {parser.parse(raw2, 'add')[:40]}")
print(f"  Has code: {parser.has_code(raw1)}")

# ── Reflexion Self-Debugging ──────────────────────────────────────────────────
print("
── Reflexion Self-Debugging ──")
agent = ReflexionAgent(sandbox, max_attempts=3)
for prob in problems[:3]:
    result = agent.solve(prob)
    print(f"  [{prob.problem_id}] {prob.entry_point}: "
          f"solved={result.success}, attempts={result.n_attempts}")

# ── pass@k Estimator ──────────────────────────────────────────────────────────
print("
── pass@k Estimator ──")
n, c = 20, 15
for k in [1, 5, 10]:
    print(f"  pass@{k} (n={n}, c={c}): {pass_at_k(n, c, k):.3f}")

# ── Benchmark Evaluation ──────────────────────────────────────────────────────
print("
── Benchmark Evaluation ──")
evaluator = CodeEvaluator(sandbox, ks=[1])

def mock_code_fn(prob):
    solutions = {
        "p001": "def add(a, b):
    return a + b",
        "p002": "def is_even(n):
    return n % 2 == 0",
        "p003": "def factorial(n):
    if n == 0: return 1
    return n * factorial(n-1)",
        "p004": "def sort_list(lst):
    return sorted(lst)",
        "p005": "def longest_common_prefix(strs):
    if not strs: return ''
    p = strs[0]
    for s in strs[1:]:
        while not s.startswith(p): p = p[:-1]
    return p",
    }
    return solutions.get(prob.problem_id, "def f(): pass")

bench = evaluator.evaluate_benchmark(problems, mock_code_fn, n_samples=1)
print(f"  Benchmark: {bench.to_dict()}")
for r in bench.per_problem:
    print(f"    {r.problem_id}: pass@1={r.pass_at_1:.2f}, mean_rate={r.mean_pass_rate:.2f}")

# ── AST Analyser ──────────────────────────────────────────────────────────────
print("
── AST Code Analysis ──")
code_to_analyse = """
def factorial(n: int) -> int:
    if n == 0:
        return 1
    return n * factorial(n - 1)

def sort_list(lst: list) -> list:
    return [x for x in sorted(lst)]
"""
analyser = ASTAnalyser(code_to_analyse)
print(f"  Functions: {analyser.list_functions()}")
for fn in analyser.list_functions():
    info = analyser.analyse_function(fn)
    print(f"  {fn}: complexity={info.complexity}, "
          f"recursion={info.has_recursion}, "
          f"comprehension={info.uses_comprehension}, "
          f"branches={info.n_branches}")
    print(f"    Issues: {analyser.detect_issues()}")

print("
Code generation demo complete!")
