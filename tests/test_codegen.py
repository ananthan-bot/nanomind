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
        ok, _ = check_imports("import os
print(os.getcwd())")
        assert not ok

    def test_safe_import(self):
        ok, _ = check_imports("import math
print(math.pi)")
        assert ok   # math is not blocked


# ── CodeSandbox ───────────────────────────────────────────────────────────────

class TestCodeSandbox:
    def test_correct_code_passes(self):
        r = SANDBOX.execute("def add(a, b):
    return a + b",
                             PROBLEMS[0].test_cases, "add")
        assert r.success

    def test_wrong_code_fails(self):
        r = SANDBOX.execute("def add(a, b):
    return a - b",
                             PROBLEMS[0].test_cases, "add")
        assert not r.success

    def test_syntax_error(self):
        r = SANDBOX.execute("def add(a, b)
    return a + b",
                             PROBLEMS[0].test_cases, "add")
        assert not r.success

    def test_pass_rate_in_range(self):
        r = SANDBOX.execute("def add(a, b):
    return a + b",
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
        raw = "```python
def add(a, b):
    return a + b
```"
        code = p.parse(raw, "add")
        assert "def add" in code

    def test_parse_raw_def(self):
        p   = CodeOutputParser()
        raw = "def add(a, b):
    return a + b

Some explanation."
        code = p.parse(raw, "add")
        assert "def add" in code

    def test_has_code_true(self):
        p = CodeOutputParser()
        assert p.has_code("```python
def f(): pass
```")

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
                                  ["def add(a, b):
    return a + b"])
        assert r.pass_at_1 == 1.0

    def test_wrong_code_low_pass(self):
        ev = CodeEvaluator(SANDBOX, ks=[1])
        r  = ev.evaluate_problem(PROBLEMS[0],
                                  ["def add(a, b):
    return 0"])
        assert r.pass_at_1 < 1.0

    def test_benchmark_returns_result(self):
        ev   = CodeEvaluator(SANDBOX, ks=[1])
        probs = PROBLEMS[:2]
        bench = ev.evaluate_benchmark(
            probs,
            lambda p: "def add(a, b): return a+b
def is_even(n): return n%2==0",
        )
        assert bench.n_problems == 2


# ── ASTAnalyser ───────────────────────────────────────────────────────────────

class TestASTAnalyser:
    def test_list_functions(self):
        a = ASTAnalyser("def foo(): pass
def bar(): pass")
        assert set(a.list_functions()) == {"foo", "bar"}

    def test_analyse_complexity(self):
        code = "def f(n):
    if n > 0:
        for i in range(n):
            pass"
        a    = ASTAnalyser(code)
        info = a.analyse_function("f")
        assert info.complexity >= 2   # 1 branch + 1 loop

    def test_detect_recursion(self):
        code = "def fact(n):
    if n == 0: return 1
    return n * fact(n-1)"
        a    = ASTAnalyser(code)
        info = a.analyse_function("fact")
        assert info.has_recursion

    def test_detect_comprehension(self):
        code = "def f(lst):
    return [x*2 for x in lst]"
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


class TestSampleProblems:
    def test_count(self):
        probs = make_sample_problems()
        assert len(probs) == 5

    def test_all_have_tests(self):
        for p in make_sample_problems():
            assert len(p.test_cases) > 0


class TestBlockedOS:
    def test_os_blocked(self):
        r = SANDBOX.execute("import os
def f(): return os.getcwd()")
        assert not r.success
        assert "Blocked" in (r.error or "")


class TestPassAtKBoundaries:
    def test_k_equals_n(self):
        r = pass_at_k(10, 5, 10)
        assert 0.0 <= r <= 1.0

    def test_c_equals_n(self):
        assert pass_at_k(5, 5, 3) == 1.0

    def test_c_zero(self):
        assert pass_at_k(5, 0, 3) == 0.0


class TestASTReturn:
    def test_has_return(self):
        a = ASTAnalyser("def f(x):
    return x + 1")
        assert a.has_return_in_all_paths("f")

    def test_no_return(self):
        a = ASTAnalyser("def f(x):
    x = x + 1")
        assert not a.has_return_in_all_paths("f")


class TestExecutionResultDict:
    def test_to_dict_keys(self):
        r = SANDBOX.execute("def add(a, b): return a+b",
                             PROBLEMS[0].test_cases, "add")
        d = r.to_dict()
        for k in ("success", "pass_rate", "n_passed", "n_total"):
            assert k in d
