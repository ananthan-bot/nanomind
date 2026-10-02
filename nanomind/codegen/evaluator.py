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
