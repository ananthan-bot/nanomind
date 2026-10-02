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
    "prompt":  "def add(a: int, b: int) -> int:\n    ...",
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
        return "
".join(lines)

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
        return "
".join(lines)

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
