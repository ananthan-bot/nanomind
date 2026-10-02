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
