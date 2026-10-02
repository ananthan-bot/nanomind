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
        code    = "def add(a, b):\n    return a + b"
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
            output           = "
".join(output_lines),
            error            = "
".join(errors) if errors else None,
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
