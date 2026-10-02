"""
nanomind/codegen/reflexion.py — Reflexion: LLM self-debugging via verbal feedback.

## Reflexion (Shinn et al., 2023)

Standard code generation:
  prompt → LLM → code → test → DONE (even if wrong)

Reflexion: iterative self-repair loop
  prompt → LLM → code → test → FAIL
    → LLM reflects on failure → revised code → test → PASS ✓

The key insight: language models can use natural language
to reason about their own mistakes!

Reflection prompt:
  "Your previous code failed with this error:
   AssertionError: Expected 3, got 2
   The test was: assert add(1, 2) == 3

   Identify the bug and write a corrected implementation."

This is used in:
  - AlphaCode 2 (self-play debugging)
  - Devin (AI software engineer)
  - SWE-bench (automated bug fixing)

Reference:
  Shinn et al. (2023) "Reflexion: Language Agents with Verbal Reinforcement Learning"
  https://arxiv.org/abs/2303.11366
"""

from __future__ import annotations
from dataclasses import dataclass, field
from nanomind.codegen.problem import CodeProblem, TestCase
from nanomind.codegen.sandbox import CodeSandbox, ExecutionResult


@dataclass
class ReflexionStep:
    """One Reflexion iteration: attempt + feedback + reflection."""
    attempt_num:   int
    generated_code: str
    exec_result:   ExecutionResult
    reflection:    str | None = None
    improved_code: str | None = None


@dataclass
class ReflexionResult:
    """Full result of a Reflexion self-debugging session."""
    problem_id:  str
    final_code:  str
    success:     bool
    n_attempts:  int
    steps:       list[ReflexionStep] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """Pass rate improvement from first to last attempt."""
        if len(self.steps) < 2:
            return 0.0
        first = self.steps[0].exec_result.pass_rate
        last  = self.steps[-1].exec_result.pass_rate
        return last - first

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "success":    self.success,
            "n_attempts": self.n_attempts,
            "improvement": round(self.improvement, 3),
        }


class ReflexionAgent:
    """
    Reflexion self-debugging agent for code generation.

    Args:
        sandbox:      :class:`CodeSandbox` for execution.
        lm_fn:        Callable (prompt: str) → str. LLM for generation.
        max_attempts: Maximum debugging iterations.

    Example::

        agent  = ReflexionAgent(sandbox, lm_fn=my_llm, max_attempts=3)
        result = agent.solve(problem, initial_code="def add(a,b):\n    return a-b")
        print(f"Solved in {result.n_attempts} attempts: {result.success}")
    """

    REFLECTION_PROMPT = (
        "The following Python code failed some test cases:

"
        "```python
{code}
```

"
        "Error/Failures:
{error}

"
        "Reflect on what went wrong and write a corrected version.
"
        "Corrected code:
```python
"
    )

    GENERATION_PROMPT = (
        "Write a Python function to solve this problem:

"
        "{description}

"
        "```python
{signature}
    pass
```

"
        "Implementation:
```python
"
    )

    def __init__(
        self,
        sandbox:      CodeSandbox,
        lm_fn:        object = None,
        max_attempts: int    = 3,
    ) -> None:
        self.sandbox      = sandbox
        self.lm_fn        = lm_fn or self._mock_lm
        self.max_attempts = max_attempts

    def _mock_lm(self, prompt: str) -> str:
        """Mock LM — returns correct solutions for standard problems."""
        if "add" in prompt:
            return "```python
def add(a, b):
    return a + b
```"
        if "even" in prompt:
            return "```python
def is_even(n):
    return n % 2 == 0
```"
        if "factorial" in prompt:
            return "```python
def factorial(n):
    if n == 0: return 1
    return n * factorial(n-1)
```"
        if "sort" in prompt:
            return "```python
def sort_list(lst):
    return sorted(lst)
```"
        if "prefix" in prompt:
            return "```python
def longest_common_prefix(strs):
    if not strs: return ''
    p = strs[0]
    for s in strs[1:]:
        while not s.startswith(p):
            p = p[:-1]
    return p
```"
        return "```python
def solution(*args):
    pass
```"

    def _generate_initial(self, problem: CodeProblem) -> str:
        prompt = self.GENERATION_PROMPT.format(
            description = problem.description,
            signature   = problem.signature,
        )
        raw = self.lm_fn(prompt)
        # Extract code from output
        import re
        m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
        return m.group(1).strip() if m else raw.strip()

    def _reflect(self, code: str, error: str) -> str:
        prompt = self.REFLECTION_PROMPT.format(code=code, error=error)
        raw = self.lm_fn(prompt)
        import re
        m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
        return m.group(1).strip() if m else raw.strip()

    def solve(
        self,
        problem:       CodeProblem,
        initial_code:  str | None = None,
    ) -> ReflexionResult:
        """
        Run Reflexion self-debugging loop.

        Args:
            problem:      :class:`CodeProblem` to solve.
            initial_code: Starting code (generate if None).

        Returns:
            :class:`ReflexionResult`.
        """
        steps     = []
        test_cases = problem.visible_tests or problem.test_cases

        # Initial generation
        code = initial_code or self._generate_initial(problem)

        for attempt in range(self.max_attempts):
            result = self.sandbox.execute(
                code, test_cases=test_cases,
                entry_point=problem.entry_point,
            )
            step = ReflexionStep(
                attempt_num     = attempt,
                generated_code  = code,
                exec_result     = result,
            )

            if result.success:
                step.reflection   = "All tests passed!"
                steps.append(step)
                return ReflexionResult(
                    problem_id = problem.problem_id,
                    final_code = code,
                    success    = True,
                    n_attempts = attempt + 1,
                    steps      = steps,
                )

            # Reflect and improve
            error_msg = result.error or f"Pass rate: {result.pass_rate:.0%}"
            reflection = f"Failed {result.n_total - result.n_passed}/{result.n_total} tests"
            improved   = self._reflect(code, error_msg)
            step.reflection   = reflection
            step.improved_code = improved
            steps.append(step)
            code = improved

        # Final attempt result
        final_result = self.sandbox.execute(
            code, test_cases=test_cases, entry_point=problem.entry_point
        )
        return ReflexionResult(
            problem_id = problem.problem_id,
            final_code = code,
            success    = final_result.success,
            n_attempts = self.max_attempts,
            steps      = steps,
        )
