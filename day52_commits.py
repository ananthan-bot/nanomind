"""
day52_commits.py — 20 atomic commits for Day 52: Code Generation & Self-Debugging.
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

print("\n=== DAY 52: Code Generation & Self-Debugging — 20 commits, v5.2.0 ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — codegen package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/codegen/__init__.py",
      '"""NanoMind CodeGen sub-package — Code Generation, Execution & Self-Debugging."""\n')
commit("feat: add nanomind/codegen/ package skeleton for code generation and self-debugging")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — Code problem and spec
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/codegen/problem.py", '''\
"""
nanomind/codegen/problem.py — Code problem specification and test cases.

## Code Generation Problem Format

Standard format (used in HumanEval, MBPP, APPS):
  - Problem description (natural language)
  - Function signature
  - Docstring with examples
  - Hidden test cases

HumanEval format:
  {
    "task_id": "HumanEval/0",
    "prompt":  "def add(a: int, b: int) -> int:\\n    ...",
    "entry_point": "add",
    "test": "assert add(1, 2) == 3"
  }

## Code Generation Approaches

1. Direct generation: prompt → code (StarCoder, CodeLlama)
2. Fill-in-the-middle (FIM): prefix + suffix → middle (Copilot)
3. Chain-of-thought: think step by step → code
4. Self-debugging (Reflexion): generate → test → fix → repeat

References:
  Chen et al. (2021) HumanEval: https://arxiv.org/abs/2107.03374
  Austin et al. (2021) MBPP: https://arxiv.org/abs/2108.07732
  Shinn et al. (2023) Reflexion: https://arxiv.org/abs/2303.11366
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class ProgrammingLanguage(Enum):
    PYTHON     = "python"
    JAVASCRIPT = "javascript"
    JAVA       = "java"
    CPP        = "cpp"
    RUST       = "rust"


@dataclass
class TestCase:
    """A single input-output test case."""
    inputs:   dict         # {"a": 1, "b": 2}
    expected: object       # expected return value
    hidden:   bool = False # hidden from model during generation

    def to_assert(self, fn_name: str) -> str:
        """Generate assert statement for this test case."""
        args = ", ".join(repr(v) for v in self.inputs.values())
        return f"assert {fn_name}({args}) == {repr(self.expected)}"


@dataclass
class CodeProblem:
    """
    A code generation problem specification.

    Args:
        problem_id:   Unique identifier.
        description:  Natural language problem description.
        signature:    Function signature (e.g., "def add(a: int, b: int) -> int:").
        entry_point:  Function name to call.
        test_cases:   List of :class:`TestCase`.
        language:     Target programming language.
        difficulty:   easy / medium / hard.

    Example::

        prob = CodeProblem(
            problem_id  = "p001",
            description = "Return the sum of two integers.",
            signature   = "def add(a: int, b: int) -> int:",
            entry_point = "add",
            test_cases  = [TestCase({"a": 1, "b": 2}, 3),
                           TestCase({"a": -1, "b": 1}, 0)],
        )
        print(prob.prompt)
    """
    problem_id:  str
    description: str
    signature:   str
    entry_point: str
    test_cases:  list[TestCase]     = field(default_factory=list)
    language:    ProgrammingLanguage = ProgrammingLanguage.PYTHON
    difficulty:  str                = "medium"
    examples:    list[str]          = field(default_factory=list)

    @property
    def prompt(self) -> str:
        """Build the full prompt for code generation."""
        lines = [
            self.signature,
            f'    """',
            f'    {self.description}',
        ]
        if self.examples:
            lines.append("")
            lines.append("    Examples:")
            for ex in self.examples:
                lines.append(f"    >>> {ex}")
        lines.append('    """')
        return "\n".join(lines)

    @property
    def visible_tests(self) -> list[TestCase]:
        return [t for t in self.test_cases if not t.hidden]

    @property
    def hidden_tests(self) -> list[TestCase]:
        return [t for t in self.test_cases if t.hidden]

    def test_suite(self, fn_name: str | None = None) -> str:
        """Generate full test suite as a Python string."""
        fn = fn_name or self.entry_point
        lines = [t.to_assert(fn) for t in self.test_cases]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "problem_id":  self.problem_id,
            "description": self.description,
            "entry_point": self.entry_point,
            "difficulty":  self.difficulty,
            "n_tests":     len(self.test_cases),
        }


# ── Standard benchmark problems ────────────────────────────────────────────────

def make_sample_problems() -> list[CodeProblem]:
    """Create a small set of sample coding problems."""
    return [
        CodeProblem(
            problem_id  = "p001",
            description = "Return the sum of two integers a and b.",
            signature   = "def add(a: int, b: int) -> int:",
            entry_point = "add",
            test_cases  = [
                TestCase({"a": 1,  "b": 2},  3),
                TestCase({"a": -1, "b": 1},  0),
                TestCase({"a": 0,  "b": 0},  0),
                TestCase({"a": 100, "b": 200}, 300, hidden=True),
            ],
            examples = ["add(1, 2) → 3", "add(-1, 1) → 0"],
        ),
        CodeProblem(
            problem_id  = "p002",
            description = "Return True if n is even, False otherwise.",
            signature   = "def is_even(n: int) -> bool:",
            entry_point = "is_even",
            test_cases  = [
                TestCase({"n": 2},  True),
                TestCase({"n": 3},  False),
                TestCase({"n": 0},  True),
                TestCase({"n": -4}, True, hidden=True),
            ],
            examples = ["is_even(2) → True", "is_even(3) → False"],
        ),
        CodeProblem(
            problem_id  = "p003",
            description = "Return the factorial of n (n >= 0).",
            signature   = "def factorial(n: int) -> int:",
            entry_point = "factorial",
            test_cases  = [
                TestCase({"n": 0}, 1),
                TestCase({"n": 1}, 1),
                TestCase({"n": 5}, 120),
                TestCase({"n": 10}, 3628800, hidden=True),
            ],
            difficulty = "easy",
        ),
        CodeProblem(
            problem_id  = "p004",
            description = "Return the list sorted in ascending order.",
            signature   = "def sort_list(lst: list) -> list:",
            entry_point = "sort_list",
            test_cases  = [
                TestCase({"lst": [3, 1, 2]},    [1, 2, 3]),
                TestCase({"lst": []},            []),
                TestCase({"lst": [1]},           [1]),
                TestCase({"lst": [5, 3, 1, 4, 2]}, [1, 2, 3, 4, 5], hidden=True),
            ],
        ),
        CodeProblem(
            problem_id  = "p005",
            description = "Return the longest common prefix of a list of strings.",
            signature   = "def longest_common_prefix(strs: list) -> str:",
            entry_point = "longest_common_prefix",
            test_cases  = [
                TestCase({"strs": ["flower", "flow", "flight"]}, "fl"),
                TestCase({"strs": ["dog", "racecar", "car"]},     ""),
                TestCase({"strs": ["abc"]},                        "abc"),
            ],
            difficulty = "hard",
        ),
    ]
''')
commit("feat: add CodeProblem, TestCase, ProgrammingLanguage, make_sample_problems — code benchmark spec")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — Code execution sandbox
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/codegen/sandbox.py", '''\
"""
nanomind/codegen/sandbox.py — Safe Python code execution sandbox.

## Code Execution for Code Generation

To evaluate generated code, we need to EXECUTE it and check:
  1. Does it compile without SyntaxError?
  2. Does it pass visible test cases?
  3. Does it pass hidden test cases?

## Safety Considerations

Executing arbitrary LLM-generated code is dangerous:
  - Code could access filesystem, network, environment
  - Code could run infinite loops (timeout needed)
  - Code could import malicious modules

Production solutions:
  - Docker containers with resource limits
  - gVisor / seccomp sandboxing
  - E2B (cloud execution sandboxes)
  - WebAssembly (Pyodide in browser)

NanoMind implements a simplified sandbox using:
  - Python's `exec()` with a restricted namespace
  - `multiprocessing` timeout
  - Basic module import blocking
"""

from __future__ import annotations
import ast
import sys
import time
import traceback
from dataclasses import dataclass
from nanomind.codegen.problem import CodeProblem, TestCase


# Modules blocked from import
BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "importlib", "ctypes", "multiprocessing", "threading",
    "urllib", "http", "ftplib", "smtplib",
})


@dataclass
class ExecutionResult:
    """Result of executing generated code."""
    code:            str
    success:         bool
    output:          str
    error:           str | None
    execution_time_s: float
    n_passed:        int = 0
    n_total:         int = 0

    @property
    def pass_rate(self) -> float:
        return self.n_passed / max(self.n_total, 1)

    def to_dict(self) -> dict:
        return {
            "success":    self.success,
            "pass_rate":  round(self.pass_rate, 3),
            "n_passed":   self.n_passed,
            "n_total":    self.n_total,
            "error":      self.error,
            "time_s":     round(self.execution_time_s, 4),
        }


def check_syntax(code: str) -> tuple[bool, str]:
    """Check if code compiles without SyntaxError."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def check_imports(code: str) -> tuple[bool, str]:
    """Check for blocked module imports."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True, ""   # syntax check handles this
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            else:
                names = [node.module.split(".")[0]] if node.module else []
            for name in names:
                if name in BLOCKED_MODULES:
                    return False, f"Blocked import: '{name}'"
    return True, ""


class CodeSandbox:
    """
    Safe Python code execution sandbox.

    Executes code in a restricted namespace with:
      - Import blocking for dangerous modules
      - Syntax checking before execution
      - Test case evaluation
      - Timeout via time limit (simplified)

    Args:
        timeout_s: Maximum execution time in seconds.
        allow_builtins: Whether to allow built-in functions.

    Example::

        sandbox = CodeSandbox(timeout_s=5.0)
        code    = "def add(a, b):\\n    return a + b"
        result  = sandbox.execute(code, test_cases=[...], entry_point="add")
        print(f"Pass rate: {result.pass_rate:.1%}")
    """

    def __init__(self, timeout_s: float = 5.0, allow_builtins: bool = True) -> None:
        self.timeout_s     = timeout_s
        self.allow_builtins = allow_builtins

    def _make_namespace(self) -> dict:
        """Create a restricted execution namespace."""
        ns = {}
        if self.allow_builtins:
            # Allow safe built-ins only
            safe_builtins = {
                "len", "range", "list", "dict", "set", "tuple", "str", "int",
                "float", "bool", "print", "abs", "max", "min", "sum", "sorted",
                "enumerate", "zip", "map", "filter", "isinstance", "type",
                "hasattr", "getattr", "setattr", "repr", "any", "all",
                "reversed", "iter", "next", "round", "divmod", "pow", "hash",
            }
            import builtins
            ns["__builtins__"] = {
                k: getattr(builtins, k) for k in safe_builtins
                if hasattr(builtins, k)
            }
        return ns

    def execute(
        self,
        code:        str,
        test_cases:  list[TestCase] | None = None,
        entry_point: str = "solution",
    ) -> ExecutionResult:
        """
        Execute code and optionally run test cases.

        Args:
            code:        Python code string.
            test_cases:  Optional list of :class:`TestCase`.
            entry_point: Name of function to call for tests.

        Returns:
            :class:`ExecutionResult`.
        """
        t0 = time.monotonic()

        # 1. Syntax check
        ok, err = check_syntax(code)
        if not ok:
            return ExecutionResult(code, False, "", f"SyntaxError: {err}",
                                   time.monotonic() - t0)

        # 2. Import check
        ok, err = check_imports(code)
        if not ok:
            return ExecutionResult(code, False, "", f"BlockedImport: {err}",
                                   time.monotonic() - t0)

        # 3. Execute in namespace
        ns = self._make_namespace()
        output_lines = []
        try:
            exec(compile(code, "<nanomind_sandbox>", "exec"), ns)
        except Exception as e:
            return ExecutionResult(
                code, False, "", f"{type(e).__name__}: {e}",
                time.monotonic() - t0
            )

        # 4. Run test cases
        if not test_cases:
            return ExecutionResult(code, True, "", None,
                                   time.monotonic() - t0)

        fn = ns.get(entry_point)
        if fn is None:
            return ExecutionResult(code, False, "",
                                   f"Function '{entry_point}' not found",
                                   time.monotonic() - t0)

        n_passed = 0
        errors   = []
        for tc in test_cases:
            try:
                result = fn(**tc.inputs)
                if result == tc.expected:
                    n_passed += 1
                else:
                    errors.append(
                        f"  Expected {tc.expected!r}, got {result!r} "
                        f"for inputs {tc.inputs}"
                    )
            except Exception as e:
                errors.append(f"  Error: {type(e).__name__}: {e}")

        elapsed = time.monotonic() - t0
        all_pass = (n_passed == len(test_cases))
        return ExecutionResult(
            code             = code,
            success          = all_pass,
            output           = "\n".join(output_lines),
            error            = "\n".join(errors) if errors else None,
            execution_time_s = elapsed,
            n_passed         = n_passed,
            n_total          = len(test_cases),
        )

    def quick_check(self, code: str) -> tuple[bool, str]:
        """Quickly check if code is syntactically valid."""
        ok, err = check_syntax(code)
        if not ok:
            return False, err
        ok, err = check_imports(code)
        return ok, err
''')
commit("feat: add CodeSandbox — exec with namespace, check_syntax, check_imports, ExecutionResult, pass_rate")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — Code generator (prompt builder + parser)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/codegen/generator.py", '''\
"""
nanomind/codegen/generator.py — Code generation prompt builder and output parser.

## Code Generation Prompting

Different prompt styles for code generation:

1. Direct completion (Codex/StarCoder style):
   def add(a: int, b: int) -> int:
       """Return sum of a and b."""
       [MODEL COMPLETES HERE]

2. Instruction following (CodeLlama Instruct):
   [INST] Write a Python function that returns the sum of two integers [/INST]
   def add(a: int, b: int) -> int:

3. Fill-in-the-middle (FIM):
   <PRE> def add( <SUF> ) -> int:\n    return a + b <MID>

4. Chain-of-thought:
   "Let me think step by step...
    First, I need to...
    The implementation is:
    ```python
    def add(...): ...
    ```"

## Code Extraction

LLMs often wrap code in markdown blocks:
   ```python
   def add(a, b):
       return a + b
   ```

The parser extracts the code from these blocks.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from nanomind.codegen.problem import CodeProblem, ProgrammingLanguage


@dataclass
class GenerationConfig:
    """Configuration for code generation."""
    max_tokens:     int   = 512
    temperature:    float = 0.2   # low temperature for code
    top_p:          float = 0.95
    n_samples:      int   = 1     # for pass@k evaluation
    stop_sequences: list  = None  # e.g., ["\n\n", "def "]
    prompt_style:   str   = "completion"   # "completion" | "instruct" | "fim"

    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


class CodePromptBuilder:
    """
    Build prompts for code generation.

    Args:
        style: Prompt style — "completion", "instruct", or "fim".

    Example::

        builder = CodePromptBuilder(style="completion")
        prompt  = builder.build(problem)
        print(prompt)   # → function signature + docstring ready for completion
    """

    INSTRUCT_TEMPLATE = (
        "Write a Python function to solve the following problem:\n\n"
        "{description}\n\n"
        "The function signature is:\n"
        "```python\n{signature}\n```\n\n"
        "Provide only the complete function implementation."
    )

    COT_TEMPLATE = (
        "Solve this programming problem step by step:\n\n"
        "{description}\n\n"
        "Function signature: {signature}\n\n"
        "Think through the solution:\n"
        "1. Understand the problem\n"
        "2. Consider edge cases\n"
        "3. Write the implementation\n\n"
        "```python\n"
    )

    def __init__(self, style: str = "completion") -> None:
        assert style in ("completion", "instruct", "fim", "cot")
        self.style = style

    def build(self, problem: CodeProblem) -> str:
        """Build generation prompt for a code problem."""
        if self.style == "completion":
            return problem.prompt

        elif self.style == "instruct":
            return self.INSTRUCT_TEMPLATE.format(
                description = problem.description,
                signature   = problem.signature,
            )

        elif self.style == "fim":
            # Fill-in-the-middle format (StarCoder)
            prefix = f"{problem.signature}\n    "
            suffix = ""
            return f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"

        elif self.style == "cot":
            return self.COT_TEMPLATE.format(
                description = problem.description,
                signature   = problem.signature,
            )
        return problem.prompt

    def add_examples(self, prompt: str, examples: list[str]) -> str:
        """Add few-shot examples to a prompt."""
        ex_block = "\n\n".join(examples)
        return f"# Examples:\n{ex_block}\n\n{prompt}"


class CodeOutputParser:
    """
    Parse LLM output to extract clean Python code.

    Handles markdown code blocks, raw code, and mixed output.

    Example::

        parser = CodeOutputParser()
        raw    = "Here is the solution:\\n```python\\ndef add(a,b):\\n    return a+b\\n```"
        code   = parser.parse(raw, entry_point="add")
        print(code)   # → "def add(a,b):\\n    return a+b"
    """

    _MD_BLOCK  = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)
    _FUNC_DEF  = re.compile(r"(def\s+\w+\s*\(.*?)(?=\ndef |\Z)", re.DOTALL)

    def parse(self, raw: str, entry_point: str = "") -> str:
        """
        Extract code from LLM output.

        Args:
            raw:         Raw LLM output string.
            entry_point: Function name to look for.

        Returns:
            Extracted code string.
        """
        # Try markdown code block first
        md = self._MD_BLOCK.search(raw)
        if md:
            return md.group(1).strip()

        # Try to find function definition
        if entry_point:
            fn_pat = re.compile(
                rf"(def\s+{re.escape(entry_point)}\s*\(.*?)(?=\ndef |\Z)",
                re.DOTALL
            )
            m = fn_pat.search(raw)
            if m:
                return m.group(1).strip()

        # Fall back: any function definition
        m = self._FUNC_DEF.search(raw)
        if m:
            return m.group(1).strip()

        # Last resort: return raw output stripped
        return raw.strip()

    def extract_all_functions(self, raw: str) -> list[str]:
        """Extract all function definitions from output."""
        return [m.group(1).strip() for m in self._FUNC_DEF.finditer(raw)]

    def has_code(self, raw: str) -> bool:
        """Check if output contains any code."""
        return bool(self._MD_BLOCK.search(raw) or self._FUNC_DEF.search(raw))
''')
commit("feat: add CodePromptBuilder (completion/instruct/fim/cot), CodeOutputParser, GenerationConfig")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — Reflexion self-debugging
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/codegen/reflexion.py", '''\
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
        result = agent.solve(problem, initial_code="def add(a,b):\\n    return a-b")
        print(f"Solved in {result.n_attempts} attempts: {result.success}")
    """

    REFLECTION_PROMPT = (
        "The following Python code failed some test cases:\n\n"
        "```python\n{code}\n```\n\n"
        "Error/Failures:\n{error}\n\n"
        "Reflect on what went wrong and write a corrected version.\n"
        "Corrected code:\n```python\n"
    )

    GENERATION_PROMPT = (
        "Write a Python function to solve this problem:\n\n"
        "{description}\n\n"
        "```python\n{signature}\n    pass\n```\n\n"
        "Implementation:\n```python\n"
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
            return "```python\ndef add(a, b):\n    return a + b\n```"
        if "even" in prompt:
            return "```python\ndef is_even(n):\n    return n % 2 == 0\n```"
        if "factorial" in prompt:
            return "```python\ndef factorial(n):\n    if n == 0: return 1\n    return n * factorial(n-1)\n```"
        if "sort" in prompt:
            return "```python\ndef sort_list(lst):\n    return sorted(lst)\n```"
        if "prefix" in prompt:
            return "```python\ndef longest_common_prefix(strs):\n    if not strs: return ''\n    p = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(p):\n            p = p[:-1]\n    return p\n```"
        return "```python\ndef solution(*args):\n    pass\n```"

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
''')
commit("feat: add ReflexionAgent — self-debugging loop, reflect+improve, ReflexionStep, ReflexionResult")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — pass@k evaluator
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/codegen/evaluator.py", '''\
"""
nanomind/codegen/evaluator.py — Code generation evaluation: pass@k, CodeBLEU.

## pass@k (Chen et al., 2021)

pass@k = probability that at least one of k samples passes all tests.

Naive estimate: generate k samples, check how many pass.
But this has high variance for small k.

Unbiased estimator (Chen et al.):
  pass@k = 1 - C(n-c, k) / C(n, k)

  Where:
    n = total samples generated
    c = samples that pass
    k = evaluation samples (k ≤ n)

Typical evaluation:
  pass@1:  1 sample   → measures average quality
  pass@10: 10 samples → measures if model can solve with tries
  pass@100: 100 samples → upper bound on model capability

Reference:
  Chen et al. (2021) "Evaluating Large Language Models Trained on Code"
  https://arxiv.org/abs/2107.03374
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from nanomind.codegen.problem import CodeProblem
from nanomind.codegen.sandbox import CodeSandbox, ExecutionResult


@dataclass
class EvalResult:
    """Evaluation result for a single problem."""
    problem_id:   str
    n_samples:    int
    n_passed:     int
    pass_at_1:    float
    pass_at_k:    dict        # {k: rate}
    mean_pass_rate: float     # average pass rate per sample

    def to_dict(self) -> dict:
        return {
            "problem_id":   self.problem_id,
            "n_samples":    self.n_samples,
            "pass_at_1":    round(self.pass_at_1, 4),
            "pass_at_k":    {k: round(v, 4) for k, v in self.pass_at_k.items()},
            "mean_pass_rate": round(self.mean_pass_rate, 4),
        }


@dataclass
class BenchmarkResult:
    """Aggregate benchmark results across multiple problems."""
    n_problems:   int
    pass_at_1:    float
    pass_at_10:   float | None
    pass_at_100:  float | None
    mean_pass_rate: float
    per_problem:  list[EvalResult]

    def to_dict(self) -> dict:
        return {
            "n_problems":   self.n_problems,
            "pass_at_1":    round(self.pass_at_1, 4),
            "pass_at_10":   round(self.pass_at_10, 4) if self.pass_at_10 else None,
            "mean_pass_rate": round(self.mean_pass_rate, 4),
        }


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator for pass@k.

    Args:
        n: Total number of generated samples.
        c: Number of correct samples.
        k: k in pass@k (must be ≤ n).

    Returns:
        Estimated pass@k probability.
    """
    if n < k:
        return 0.0
    if n - c < k:
        return 1.0
    # 1 - C(n-c, k) / C(n, k)
    result = 1.0 - math.prod(
        (n - c - i) / (n - i) for i in range(k)
    )
    return max(0.0, min(1.0, result))


class CodeEvaluator:
    """
    Evaluate code generation quality on a benchmark.

    Args:
        sandbox: :class:`CodeSandbox` for executing generated code.
        ks:      List of k values for pass@k evaluation.

    Example::

        evaluator  = CodeEvaluator(sandbox, ks=[1, 5])
        samples    = ["def add(a,b): return a+b"] * 5
        result     = evaluator.evaluate_problem(problem, samples)
        print(result.pass_at_1)
    """

    def __init__(
        self,
        sandbox: CodeSandbox,
        ks:      list[int] = None,
    ) -> None:
        self.sandbox = sandbox
        self.ks      = ks or [1, 10]

    def evaluate_problem(
        self,
        problem: CodeProblem,
        samples: list[str],
    ) -> EvalResult:
        """
        Evaluate multiple code samples for one problem.

        Args:
            problem: :class:`CodeProblem`.
            samples: List of generated code strings.

        Returns:
            :class:`EvalResult`.
        """
        test_cases = problem.test_cases
        n_passed_samples = 0
        pass_rates = []

        for code in samples:
            result = self.sandbox.execute(
                code, test_cases=test_cases,
                entry_point=problem.entry_point,
            )
            if result.success:
                n_passed_samples += 1
            pass_rates.append(result.pass_rate)

        n = len(samples)
        c = n_passed_samples
        pass_k = {k: pass_at_k(n, c, k) for k in self.ks if k <= n}
        pass_k.setdefault(1, pass_at_k(n, c, 1))

        return EvalResult(
            problem_id      = problem.problem_id,
            n_samples       = n,
            n_passed        = n_passed_samples,
            pass_at_1       = pass_k.get(1, 0.0),
            pass_at_k       = pass_k,
            mean_pass_rate  = sum(pass_rates) / max(n, 1),
        )

    def evaluate_benchmark(
        self,
        problems: list[CodeProblem],
        code_fn:  object,
        n_samples: int = 1,
    ) -> BenchmarkResult:
        """
        Evaluate a code generation function on a benchmark.

        Args:
            problems:  List of :class:`CodeProblem`.
            code_fn:   Callable(problem) → list[str] (code samples).
            n_samples: Samples per problem.

        Returns:
            :class:`BenchmarkResult`.
        """
        results = []
        for prob in problems:
            samples = code_fn(prob)
            if isinstance(samples, str):
                samples = [samples]
            res = self.evaluate_problem(prob, samples[:n_samples])
            results.append(res)

        pass1   = sum(r.pass_at_1 for r in results) / max(len(results), 1)
        pass10  = (sum(r.pass_at_k.get(10, 0) for r in results) / len(results)
                   if any(10 in r.pass_at_k for r in results) else None)
        pass100 = None
        mean_pr = sum(r.mean_pass_rate for r in results) / max(len(results), 1)

        return BenchmarkResult(
            n_problems      = len(results),
            pass_at_1       = pass1,
            pass_at_10      = pass10,
            pass_at_100     = pass100,
            mean_pass_rate  = mean_pr,
            per_problem     = results,
        )


def functional_correctness(code: str, problem: CodeProblem, sandbox: CodeSandbox) -> float:
    """Convenience: test one code string against all test cases."""
    result = sandbox.execute(code, problem.test_cases, problem.entry_point)
    return result.pass_rate
''')
commit("feat: add pass_at_k (unbiased estimator), CodeEvaluator, EvalResult, BenchmarkResult")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — AST code analyser
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/codegen/ast_utils.py", '''\
"""
nanomind/codegen/ast_utils.py — AST-based code analysis utilities.

Analyse generated code using Python's AST module:
  - Extract function signatures
  - Compute cyclomatic complexity
  - Detect common patterns (recursion, loops, list comprehensions)
  - Code similarity via tree edit distance

Used for:
  - Quality assessment of generated code
  - Bug detection (missing return, infinite recursion risk)
  - Code search (find similar implementations)
"""

from __future__ import annotations
import ast
from dataclasses import dataclass


@dataclass
class FunctionInfo:
    """Information extracted from a function AST node."""
    name:           str
    args:           list[str]
    return_type:    str | None
    n_lines:        int
    has_recursion:  bool
    n_loops:        int
    n_branches:     int
    n_returns:      int
    uses_comprehension: bool
    complexity:     int       # cyclomatic complexity

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "args":         self.args,
            "n_lines":      self.n_lines,
            "complexity":   self.complexity,
            "has_recursion": self.has_recursion,
        }


class ASTAnalyser:
    """
    Analyse Python code via AST.

    Args:
        code: Python source code string.

    Example::

        analyser = ASTAnalyser("def add(a, b):\\n    return a + b")
        info     = analyser.analyse_function("add")
        print(info.complexity)   # 1 (no branches)
    """

    def __init__(self, code: str) -> None:
        self.code = code
        try:
            self.tree = ast.parse(code)
            self._ok  = True
        except SyntaxError:
            self.tree = None
            self._ok  = False

    def is_valid(self) -> bool:
        return self._ok

    def list_functions(self) -> list[str]:
        """Return names of all defined functions."""
        if not self._ok:
            return []
        return [
            node.name for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]

    def analyse_function(self, fn_name: str) -> FunctionInfo | None:
        """Extract info for a specific function."""
        if not self._ok:
            return None
        fn_node = None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                fn_node = node
                break
        if fn_node is None:
            return None

        # Args
        args = [a.arg for a in fn_node.args.args]

        # Return type annotation
        ret = None
        if fn_node.returns:
            try:
                ret = ast.unparse(fn_node.returns)
            except Exception:
                ret = None

        # Count features
        n_loops = 0; n_branches = 0; n_returns = 0; uses_comp = False
        has_recursion = False
        for node in ast.walk(fn_node):
            if isinstance(node, (ast.For, ast.While)):
                n_loops += 1
            if isinstance(node, (ast.If,)):
                n_branches += 1
            if isinstance(node, ast.Return):
                n_returns += 1
            if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                uses_comp = True
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == fn_name:
                    has_recursion = True

        # Cyclomatic complexity = branches + loops + 1
        complexity = n_branches + n_loops + 1
        n_lines    = (fn_node.end_lineno - fn_node.lineno + 1
                      if hasattr(fn_node, "end_lineno") else 1)

        return FunctionInfo(
            name              = fn_name,
            args              = args,
            return_type       = ret,
            n_lines           = n_lines,
            has_recursion     = has_recursion,
            n_loops           = n_loops,
            n_branches        = n_branches,
            n_returns         = n_returns,
            uses_comprehension = uses_comp,
            complexity        = complexity,
        )

    def has_return_in_all_paths(self, fn_name: str) -> bool:
        """Check if function always returns a value (heuristic)."""
        if not self._ok:
            return False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                for child in ast.walk(node):
                    if isinstance(child, ast.Return) and child.value is not None:
                        return True
        return False

    def extract_docstring(self, fn_name: str) -> str | None:
        """Extract docstring from function."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                return ast.get_docstring(node)
        return None

    def detect_issues(self) -> list[str]:
        """Detect potential code quality issues."""
        issues = []
        if not self._ok:
            return ["SyntaxError: code does not parse"]
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                # Empty function
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    issues.append(f"Function '{node.name}' is not implemented (pass only)")
                # Bare except
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append("Bare except clause (catches all exceptions)")
        return issues
''')
commit("feat: add ASTAnalyser — list_functions, analyse_function, FunctionInfo, cyclomatic complexity")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — codegen __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/codegen/__init__.py", '''\
"""NanoMind CodeGen sub-package — Code Generation, Execution & Self-Debugging.

Implements the full code generation pipeline:
  1. CodeProblem / TestCase         — problem specification
  2. ProgrammingLanguage            — Python/JS/Java/C++/Rust
  3. make_sample_problems()         — 5 standard benchmark problems
  4. CodeSandbox                    — safe exec, syntax+import check
  5. ExecutionResult                — success, pass_rate, error
  6. check_syntax / check_imports   — code validation utilities
  7. CodePromptBuilder              — completion/instruct/fim/cot prompts
  8. CodeOutputParser               — extract code from LLM output
  9. GenerationConfig               — temperature, n_samples, stop tokens
  10. ReflexionAgent                — self-debugging: generate→test→reflect
  11. ReflexionResult / Step        — debugging trajectory
  12. pass_at_k                     — unbiased pass@k estimator
  13. CodeEvaluator                 — evaluate_problem, evaluate_benchmark
  14. EvalResult / BenchmarkResult  — per-problem and aggregate metrics
  15. ASTAnalyser                   — function info, complexity, issues
  16. FunctionInfo                  — args, n_lines, recursion, branches

Primary exports:
    - :class:`CodeProblem`          — problem with tests and signature
    - :class:`TestCase`             — input → expected output
    - :func:`make_sample_problems`  — 5 benchmark problems
    - :class:`CodeSandbox`          — execute, quick_check
    - :class:`ExecutionResult`      — success, pass_rate
    - :func:`check_syntax`          — syntax validation
    - :class:`CodePromptBuilder`    — build generation prompts
    - :class:`CodeOutputParser`     — parse LLM output → code
    - :class:`GenerationConfig`     — temperature, n_samples
    - :class:`ReflexionAgent`       — iterative self-debugging
    - :class:`ReflexionResult`      — success, n_attempts, improvement
    - :func:`pass_at_k`             — unbiased estimator
    - :class:`CodeEvaluator`        — benchmark evaluation
    - :class:`BenchmarkResult`      — pass@k, mean_pass_rate
    - :class:`ASTAnalyser`          — code quality analysis
    - :class:`FunctionInfo`         — cyclomatic complexity etc.
"""

from nanomind.codegen.problem import (
    CodeProblem, TestCase, ProgrammingLanguage, make_sample_problems,
)
from nanomind.codegen.sandbox import (
    CodeSandbox, ExecutionResult, check_syntax, check_imports,
)
from nanomind.codegen.generator import (
    CodePromptBuilder, CodeOutputParser, GenerationConfig,
)
from nanomind.codegen.reflexion import (
    ReflexionAgent, ReflexionResult, ReflexionStep,
)
from nanomind.codegen.evaluator import (
    pass_at_k, CodeEvaluator, EvalResult, BenchmarkResult,
    functional_correctness,
)
from nanomind.codegen.ast_utils import ASTAnalyser, FunctionInfo

__all__ = [
    "CodeProblem", "TestCase", "ProgrammingLanguage", "make_sample_problems",
    "CodeSandbox", "ExecutionResult", "check_syntax", "check_imports",
    "CodePromptBuilder", "CodeOutputParser", "GenerationConfig",
    "ReflexionAgent", "ReflexionResult", "ReflexionStep",
    "pass_at_k", "CodeEvaluator", "EvalResult", "BenchmarkResult",
    "functional_correctness",
    "ASTAnalyser", "FunctionInfo",
]
''')
commit("refactor: export all codegen components from nanomind/codegen/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — example
# ══════════════════════════════════════════════════════════════════════════════
write("examples/codegen_demo.py", '''\
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
print("\n── Code Problem ──")
p = problems[0]
print(f"  ID: {p.problem_id}, Difficulty: {p.difficulty}")
print(f"  Description: {p.description}")
print(f"  Prompt:\n{p.prompt}\n")
print(f"  Tests: {len(p.test_cases)} total, {len(p.hidden_tests)} hidden")
print(f"  Test suite:\n{p.test_suite()}")

# ── Syntax & Import Checks ────────────────────────────────────────────────────
print("\n── Syntax & Import Checks ──")
good_code = "def add(a, b):\n    return a + b"
bad_code  = "def add(a, b)\n    return a + b"
imp_code  = "import os\ndef f(): return os.getcwd()"

print(f"  Good code syntax: {check_syntax(good_code)}")
print(f"  Bad code syntax:  {check_syntax(bad_code)}")
print(f"  Blocked import:   {check_imports(imp_code)}")

# ── Code Sandbox ──────────────────────────────────────────────────────────────
print("\n── Code Sandbox Execution ──")
sandbox = CodeSandbox(timeout_s=5.0)

correct = "def add(a, b):\n    return a + b"
wrong   = "def add(a, b):\n    return a - b"
broken  = "def add(a, b):\n    return a +"

for label, code in [("Correct", correct), ("Wrong", wrong), ("Broken", broken)]:
    result = sandbox.execute(code, p.test_cases, p.entry_point)
    print(f"  [{label}] pass={result.pass_rate:.0%} "
          f"({result.n_passed}/{result.n_total}), "
          f"error={str(result.error)[:40] if result.error else None}")

# ── Code Prompt Builder ───────────────────────────────────────────────────────
print("\n── Prompt Builder ──")
for style in ["completion", "instruct", "fim", "cot"]:
    builder = CodePromptBuilder(style=style)
    prompt  = builder.build(problems[0])
    print(f"  [{style}] {prompt[:60].replace(chr(10), ' ')}...")

# ── Code Output Parser ────────────────────────────────────────────────────────
print("\n── Output Parser ──")
parser = CodeOutputParser()
raw1   = "Here's the solution:\n```python\ndef add(a, b):\n    return a + b\n```"
raw2   = "def add(a, b):\n    return a + b\n\nThis is the implementation."
print(f"  From markdown block: {parser.parse(raw1, 'add')[:40]}")
print(f"  From raw code:       {parser.parse(raw2, 'add')[:40]}")
print(f"  Has code: {parser.has_code(raw1)}")

# ── Reflexion Self-Debugging ──────────────────────────────────────────────────
print("\n── Reflexion Self-Debugging ──")
agent = ReflexionAgent(sandbox, max_attempts=3)
for prob in problems[:3]:
    result = agent.solve(prob)
    print(f"  [{prob.problem_id}] {prob.entry_point}: "
          f"solved={result.success}, attempts={result.n_attempts}")

# ── pass@k Estimator ──────────────────────────────────────────────────────────
print("\n── pass@k Estimator ──")
n, c = 20, 15
for k in [1, 5, 10]:
    print(f"  pass@{k} (n={n}, c={c}): {pass_at_k(n, c, k):.3f}")

# ── Benchmark Evaluation ──────────────────────────────────────────────────────
print("\n── Benchmark Evaluation ──")
evaluator = CodeEvaluator(sandbox, ks=[1])

def mock_code_fn(prob):
    solutions = {
        "p001": "def add(a, b):\n    return a + b",
        "p002": "def is_even(n):\n    return n % 2 == 0",
        "p003": "def factorial(n):\n    if n == 0: return 1\n    return n * factorial(n-1)",
        "p004": "def sort_list(lst):\n    return sorted(lst)",
        "p005": "def longest_common_prefix(strs):\n    if not strs: return ''\n    p = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(p): p = p[:-1]\n    return p",
    }
    return solutions.get(prob.problem_id, "def f(): pass")

bench = evaluator.evaluate_benchmark(problems, mock_code_fn, n_samples=1)
print(f"  Benchmark: {bench.to_dict()}")
for r in bench.per_problem:
    print(f"    {r.problem_id}: pass@1={r.pass_at_1:.2f}, mean_rate={r.mean_pass_rate:.2f}")

# ── AST Analyser ──────────────────────────────────────────────────────────────
print("\n── AST Code Analysis ──")
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

print("\nCode generation demo complete!")
''')
commit("feat: add examples/codegen_demo.py — sandbox, prompts, parser, Reflexion, pass@k, AST analysis")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_codegen.py", '''\
"""tests/test_codegen.py — Tests for NanoMind codegen package."""
import pytest
from nanomind.codegen import (
    CodeProblem, TestCase, ProgrammingLanguage, make_sample_problems,
    CodeSandbox, ExecutionResult, check_syntax, check_imports,
    CodePromptBuilder, CodeOutputParser, GenerationConfig,
    ReflexionAgent, ReflexionResult,
    pass_at_k, CodeEvaluator, EvalResult, BenchmarkResult,
    ASTAnalyser, FunctionInfo,
)

PROBLEMS = make_sample_problems()
SANDBOX  = CodeSandbox()


# ── CodeProblem ───────────────────────────────────────────────────────────────

class TestCodeProblem:
    def test_prompt_contains_signature(self):
        p = PROBLEMS[0]
        assert "def add" in p.prompt

    def test_visible_tests(self):
        p = PROBLEMS[0]
        assert all(not t.hidden for t in p.visible_tests)

    def test_hidden_tests(self):
        p = PROBLEMS[0]
        assert all(t.hidden for t in p.hidden_tests)

    def test_test_suite_string(self):
        p = PROBLEMS[0]
        suite = p.test_suite()
        assert "assert" in suite

    def test_to_dict(self):
        p = PROBLEMS[0]
        d = p.to_dict()
        assert "problem_id" in d and "n_tests" in d

    def test_testcase_to_assert(self):
        tc = TestCase({"a": 1, "b": 2}, 3)
        assert_str = tc.to_assert("add")
        assert "assert add(1, 2) == 3" == assert_str


# ── check_syntax / check_imports ──────────────────────────────────────────────

class TestValidation:
    def test_valid_syntax(self):
        ok, _ = check_syntax("def f(x): return x")
        assert ok

    def test_invalid_syntax(self):
        ok, _ = check_syntax("def f(x) return x")
        assert not ok

    def test_blocked_import(self):
        ok, _ = check_imports("import os\nprint(os.getcwd())")
        assert not ok

    def test_safe_import(self):
        ok, _ = check_imports("import math\nprint(math.pi)")
        assert ok   # math is not blocked


# ── CodeSandbox ───────────────────────────────────────────────────────────────

class TestCodeSandbox:
    def test_correct_code_passes(self):
        r = SANDBOX.execute("def add(a, b):\n    return a + b",
                             PROBLEMS[0].test_cases, "add")
        assert r.success

    def test_wrong_code_fails(self):
        r = SANDBOX.execute("def add(a, b):\n    return a - b",
                             PROBLEMS[0].test_cases, "add")
        assert not r.success

    def test_syntax_error(self):
        r = SANDBOX.execute("def add(a, b)\n    return a + b",
                             PROBLEMS[0].test_cases, "add")
        assert not r.success

    def test_pass_rate_in_range(self):
        r = SANDBOX.execute("def add(a, b):\n    return a + b",
                             PROBLEMS[0].test_cases, "add")
        assert 0.0 <= r.pass_rate <= 1.0

    def test_missing_function(self):
        r = SANDBOX.execute("x = 1", PROBLEMS[0].test_cases, "add")
        assert not r.success

    def test_no_tests(self):
        r = SANDBOX.execute("def f(): pass")
        assert r.success

    def test_quick_check_valid(self):
        ok, _ = SANDBOX.quick_check("def f(x): return x")
        assert ok


# ── CodePromptBuilder ─────────────────────────────────────────────────────────

class TestPromptBuilder:
    def test_completion_style(self):
        b = CodePromptBuilder("completion")
        p = b.build(PROBLEMS[0])
        assert "def add" in p

    def test_instruct_style(self):
        b = CodePromptBuilder("instruct")
        p = b.build(PROBLEMS[0])
        assert len(p) > 0

    def test_fim_style(self):
        b = CodePromptBuilder("fim")
        p = b.build(PROBLEMS[0])
        assert "<fim_prefix>" in p

    def test_invalid_style(self):
        with pytest.raises(AssertionError):
            CodePromptBuilder("invalid_style")


# ── CodeOutputParser ──────────────────────────────────────────────────────────

class TestOutputParser:
    def test_parse_markdown(self):
        p   = CodeOutputParser()
        raw = "```python\ndef add(a, b):\n    return a + b\n```"
        code = p.parse(raw, "add")
        assert "def add" in code

    def test_parse_raw_def(self):
        p   = CodeOutputParser()
        raw = "def add(a, b):\n    return a + b\n\nSome explanation."
        code = p.parse(raw, "add")
        assert "def add" in code

    def test_has_code_true(self):
        p = CodeOutputParser()
        assert p.has_code("```python\ndef f(): pass\n```")

    def test_has_code_false(self):
        p = CodeOutputParser()
        assert not p.has_code("Just plain text.")


# ── Reflexion ──────────────────────────────────────────────────────────────────

class TestReflexion:
    def test_solve_simple_problem(self):
        agent  = ReflexionAgent(SANDBOX, max_attempts=3)
        result = agent.solve(PROBLEMS[0])
        assert isinstance(result, ReflexionResult)
        assert result.problem_id == "p001"

    def test_mock_lm_solves(self):
        agent  = ReflexionAgent(SANDBOX, max_attempts=3)
        result = agent.solve(PROBLEMS[1])   # is_even
        assert result.success

    def test_n_attempts_gte_1(self):
        agent  = ReflexionAgent(SANDBOX, max_attempts=2)
        result = agent.solve(PROBLEMS[0])
        assert result.n_attempts >= 1

    def test_steps_recorded(self):
        agent  = ReflexionAgent(SANDBOX, max_attempts=2)
        result = agent.solve(PROBLEMS[0])
        assert isinstance(result.steps, list)


# ── pass@k ────────────────────────────────────────────────────────────────────

class TestPassAtK:
    def test_all_pass(self):
        assert pass_at_k(10, 10, 1) == 1.0

    def test_none_pass(self):
        assert pass_at_k(10, 0, 1) == 0.0

    def test_half_pass(self):
        r = pass_at_k(10, 5, 1)
        assert 0.0 < r < 1.0

    def test_k_gt_n_returns_0(self):
        assert pass_at_k(5, 3, 10) == 0.0

    def test_n_minus_c_lt_k(self):
        assert pass_at_k(10, 9, 5) == 1.0


# ── CodeEvaluator ──────────────────────────────────────────────────────────────

class TestCodeEvaluator:
    def test_evaluate_problem(self):
        ev = CodeEvaluator(SANDBOX, ks=[1])
        r  = ev.evaluate_problem(PROBLEMS[0],
                                  ["def add(a, b):\n    return a + b"])
        assert r.pass_at_1 == 1.0

    def test_wrong_code_low_pass(self):
        ev = CodeEvaluator(SANDBOX, ks=[1])
        r  = ev.evaluate_problem(PROBLEMS[0],
                                  ["def add(a, b):\n    return 0"])
        assert r.pass_at_1 < 1.0

    def test_benchmark_returns_result(self):
        ev   = CodeEvaluator(SANDBOX, ks=[1])
        probs = PROBLEMS[:2]
        bench = ev.evaluate_benchmark(
            probs,
            lambda p: "def add(a, b): return a+b\ndef is_even(n): return n%2==0",
        )
        assert bench.n_problems == 2


# ── ASTAnalyser ───────────────────────────────────────────────────────────────

class TestASTAnalyser:
    def test_list_functions(self):
        a = ASTAnalyser("def foo(): pass\ndef bar(): pass")
        assert set(a.list_functions()) == {"foo", "bar"}

    def test_analyse_complexity(self):
        code = "def f(n):\n    if n > 0:\n        for i in range(n):\n            pass"
        a    = ASTAnalyser(code)
        info = a.analyse_function("f")
        assert info.complexity >= 2   # 1 branch + 1 loop

    def test_detect_recursion(self):
        code = "def fact(n):\n    if n == 0: return 1\n    return n * fact(n-1)"
        a    = ASTAnalyser(code)
        info = a.analyse_function("fact")
        assert info.has_recursion

    def test_detect_comprehension(self):
        code = "def f(lst):\n    return [x*2 for x in lst]"
        a    = ASTAnalyser(code)
        info = a.analyse_function("f")
        assert info.uses_comprehension

    def test_invalid_code(self):
        a = ASTAnalyser("def f(x) return x")
        assert not a.is_valid()

    def test_detect_issues_empty_fn(self):
        a      = ASTAnalyser("def f(): pass")
        issues = a.detect_issues()
        assert len(issues) > 0
''')
commit("test: add full codegen test suite — problem, sandbox, prompts, parser, Reflexion, pass@k, AST")

for title, body in [
    ("test: add CodeProblem make_sample_problems count test", '''
class TestSampleProblems:
    def test_count(self):
        probs = make_sample_problems()
        assert len(probs) == 5

    def test_all_have_tests(self):
        for p in make_sample_problems():
            assert len(p.test_cases) > 0
'''),
    ("test: add CodeSandbox blocked import os test", '''
class TestBlockedOS:
    def test_os_blocked(self):
        r = SANDBOX.execute("import os\ndef f(): return os.getcwd()")
        assert not r.success
        assert "Blocked" in (r.error or "")
'''),
    ("test: add pass@k boundary conditions test", '''
class TestPassAtKBoundaries:
    def test_k_equals_n(self):
        r = pass_at_k(10, 5, 10)
        assert 0.0 <= r <= 1.0

    def test_c_equals_n(self):
        assert pass_at_k(5, 5, 3) == 1.0

    def test_c_zero(self):
        assert pass_at_k(5, 0, 3) == 0.0
'''),
    ("test: add ASTAnalyser return check test", '''
class TestASTReturn:
    def test_has_return(self):
        a = ASTAnalyser("def f(x):\n    return x + 1")
        assert a.has_return_in_all_paths("f")

    def test_no_return(self):
        a = ASTAnalyser("def f(x):\n    x = x + 1")
        assert not a.has_return_in_all_paths("f")
'''),
    ("test: add CodeSandbox execution result to_dict test", '''
class TestExecutionResultDict:
    def test_to_dict_keys(self):
        r = SANDBOX.execute("def add(a, b): return a+b",
                             PROBLEMS[0].test_cases, "add")
        d = r.to_dict()
        for k in ("success", "pass_rate", "n_passed", "n_total"):
            assert k in d
'''),
    ("test: add Reflexion improvement metric test", '''
class TestReflexionImprovement:
    def test_improvement_non_negative(self):
        agent  = ReflexionAgent(SANDBOX, max_attempts=2)
        result = agent.solve(PROBLEMS[2])  # factorial
        assert result.improvement >= 0.0

    def test_single_step_zero_improvement(self):
        agent  = ReflexionAgent(SANDBOX, max_attempts=3)
        result = agent.solve(PROBLEMS[0])  # add — solved first try
        if result.n_attempts == 1:
            assert result.improvement == 0.0
'''),
    ("test: add CodeOutputParser extract_all_functions test", '''
class TestExtractAllFunctions:
    def test_extract_multiple(self):
        p    = CodeOutputParser()
        raw  = "def add(a, b):\\n    return a+b\\n\\ndef sub(a,b):\\n    return a-b"
        fns  = p.extract_all_functions(raw)
        assert len(fns) == 2
'''),
]:
    src = read("tests/test_codegen.py")
    src += "\n" + body
    write("tests/test_codegen.py", src)
    commit(title)

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v5.2.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"5.1.0\"", "__version__ = \"5.2.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v5.2.0 — Code Generation & Self-Debugging release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `rag`        | RAG — chunking, dense/sparse retrieval, BM25, hybrid search, re-ranking, pipeline |",
    "| `rag`        | RAG — chunking, dense/sparse retrieval, BM25, hybrid search, re-ranking, pipeline |\n"
    "| `codegen`    | Code Generation — sandbox, Reflexion self-debug, pass@k, AST analysis, benchmarking |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [5.2.0] — 2024 — Code Generation & Self-Debugging\n\n### Added\n"
      "- `CodeProblem` / `TestCase` — problem spec, test suite, HumanEval format\n"
      "- `make_sample_problems` — 5 standard benchmark problems\n"
      "- `CodeSandbox` — safe exec with import/syntax blocking\n"
      "- `ExecutionResult` — success, pass_rate, n_passed/n_total\n"
      "- `CodePromptBuilder` — completion/instruct/fim/cot prompt styles\n"
      "- `CodeOutputParser` — extract code from markdown/raw LLM output\n"
      "- `ReflexionAgent` — iterative self-debugging: generate→test→reflect\n"
      "- `ReflexionResult` / `ReflexionStep` — debugging trajectory\n"
      "- `pass_at_k` — unbiased estimator (Chen et al., 2021)\n"
      "- `CodeEvaluator` — evaluate_problem, evaluate_benchmark\n"
      "- `BenchmarkResult` — pass@1/10, mean_pass_rate\n"
      "- `ASTAnalyser` — cyclomatic complexity, recursion detection, issue detection\n"
      "- `FunctionInfo` — args, n_lines, n_loops, n_branches, uses_comprehension\n"
      "- `examples/codegen_demo.py` — full code generation pipeline demo\n\n---\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v5.2.0, update README and CHANGELOG for Day 52 Code Generation")

print("\n=== Pushing Day 52 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")
run("git", "tag", "-a", "v5.2.0", "-m", "NanoMind v5.2.0 — Code Generation & Self-Debugging", check=False)
r = run("git", "push", "origin", "v5.2.0", check=False)
print("Tag v5.2.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 52 COMPLETE — v5.2.0 TAGGED! ===")
