"""tests/test_eval.py — Tests for NanoMind eval package."""
import pytest
import torch
import torch.nn as nn
from nanomind.eval import (
    EvalSample, EvalTask, TaskType, EvalProtocol,
    make_mmlu_task, make_gsm8k_task, make_hellaswag_task, make_truthfulqa_task,
    exact_match, token_f1, rouge_l, math_exact_match,
    multiple_choice_accuracy, generation_metrics, perplexity,
    LLMJudge, JudgeResult, PairwiseResult,
    BenchmarkEvaluator, SampleResult, TaskResult, BenchmarkReport,
    Leaderboard, ModelScore,
)

V = 16

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 8)
        self.rnn  = nn.GRU(8, 16, batch_first=True)
        self.head = nn.Linear(16, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x % V))
        return self.head(h), None


# ── EvalSample / EvalTask ──────────────────────────────────────────────────────

class TestEvalSample:
    def test_answer_text_mc(self):
        s = EvalSample("id", "Q?", "B", TaskType.MULTIPLE_CHOICE,
                        choices=["X", "Y", "Z"])
        assert s.answer_text == "Y"

    def test_format_choices(self):
        s = EvalSample("id", "Q?", "A", choices=["opt1", "opt2"])
        fmt = s.format_choices()
        assert "A. opt1" in fmt and "B. opt2" in fmt

    def test_to_dict(self):
        s = EvalSample("id1", "Q?", "A")
        d = s.to_dict()
        assert "sample_id" in d and "answer" in d


class TestEvalTask:
    def test_len(self):
        t = make_mmlu_task(n=5)
        assert len(t) == 5

    def test_subjects(self):
        t = make_mmlu_task(n=10)
        assert len(t.subjects()) > 0

    def test_by_subject(self):
        t = make_mmlu_task(n=10)
        d = t.by_subject()
        assert all(isinstance(v, list) for v in d.values())


# ── Benchmark Generators ──────────────────────────────────────────────────────

class TestBenchmarkGenerators:
    def test_mmlu(self):
        t = make_mmlu_task(n=5)
        assert t.name == "MMLU"
        assert all(s.task_type == TaskType.MULTIPLE_CHOICE for s in t.samples)

    def test_gsm8k(self):
        t = make_gsm8k_task(n=4)
        assert t.name == "GSM8K"
        assert all(s.task_type == TaskType.MATH_REASONING for s in t.samples)

    def test_hellaswag(self):
        t = make_hellaswag_task(n=3)
        assert t.name == "HellaSwag"

    def test_truthfulqa(self):
        t = make_truthfulqa_task(n=3)
        assert t.name == "TruthfulQA"


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_exact_match_true(self):
        assert exact_match("Paris", "Paris") is True

    def test_exact_match_false(self):
        assert exact_match("Paris", "London") is False

    def test_exact_match_normalized(self):
        assert exact_match("PARIS.", "paris") is True

    def test_token_f1_perfect(self):
        assert abs(token_f1("hello world", "hello world") - 1.0) < 0.01

    def test_token_f1_zero(self):
        assert token_f1("abc", "xyz") == 0.0

    def test_token_f1_partial(self):
        f1 = token_f1("the cat sat", "cat sat mat")
        assert 0.0 < f1 < 1.0

    def test_rouge_l_perfect(self):
        assert abs(rouge_l("the cat", "the cat") - 1.0) < 0.01

    def test_rouge_l_zero(self):
        assert rouge_l("abc", "xyz") == 0.0

    def test_math_exact_match(self):
        assert math_exact_match("The answer is 42", "42") is True

    def test_math_exact_match_float(self):
        assert math_exact_match("Result: 3.14", "3.14") is True

    def test_mc_accuracy(self):
        r = multiple_choice_accuracy(["A", "B", "C"], ["A", "B", "A"])
        assert abs(r.value - 2/3) < 0.01

    def test_mc_accuracy_perfect(self):
        r = multiple_choice_accuracy(["A", "B"], ["A", "B"])
        assert r.value == 1.0

    def test_perplexity_positive(self):
        m   = TinyLM()
        ids = torch.randint(0, V, (1, 16))
        ppl = perplexity(m, ids)
        assert ppl > 0.0


# ── LLMJudge ──────────────────────────────────────────────────────────────────

class TestLLMJudge:
    def test_score_returns_result(self):
        j = LLMJudge()
        r = j.score_response("Q?", "A good response")
        assert isinstance(r, JudgeResult)
        assert 1.0 <= r.score <= 10.0

    def test_normalized_score_range(self):
        j = LLMJudge()
        r = j.score_response("Q?", "response")
        assert 0.0 <= r.normalized_score <= 1.0

    def test_compare_returns_winner(self):
        j    = LLMJudge()
        pair = j.compare("Q?", "Good answer", "Bad answer")
        assert pair.winner in ("A", "B", "tie")

    def test_win_rate(self):
        j = LLMJudge()
        wr = j.win_rate(["Q1", "Q2"], ["A1", "A2"], ["B1", "B2"])
        assert "win_rate_a" in wr
        assert abs(wr["win_rate_a"] + wr["win_rate_b"] + wr["tie_rate"] - 1.0) < 0.01

    def test_batch_score(self):
        j = LLMJudge()
        results = j.batch_score(["Q1", "Q2"], ["A1", "A2"])
        assert len(results) == 2


# ── BenchmarkEvaluator ────────────────────────────────────────────────────────

class TestBenchmarkEvaluator:
    def _evaluator(self):
        def fn(s):
            return s.answer   # oracle
        return BenchmarkEvaluator(fn, "oracle")

    def test_evaluate_task_accuracy(self):
        ev   = self._evaluator()
        task = make_mmlu_task(n=5)
        r    = ev.evaluate_task(task)
        assert 0.0 <= r.accuracy <= 1.0

    def test_oracle_gets_perfect_gsm8k(self):
        ev   = self._evaluator()
        task = make_gsm8k_task(n=4)
        r    = ev.evaluate_task(task)
        assert r.accuracy == 1.0

    def test_evaluate_all_returns_report(self):
        ev     = self._evaluator()
        tasks  = [make_mmlu_task(5), make_hellaswag_task(3)]
        report = ev.evaluate_all(tasks)
        assert isinstance(report, BenchmarkReport)
        assert len(report.tasks) == 2

    def test_by_subject(self):
        ev = self._evaluator()
        r  = ev.evaluate_task(make_mmlu_task(n=10))
        assert len(r.by_subject) > 0

    def test_n_correct(self):
        ev = self._evaluator()
        r  = ev.evaluate_task(make_gsm8k_task(n=4))
        assert r.n_correct == r.n_samples   # oracle is perfect


# ── Leaderboard ───────────────────────────────────────────────────────────────

class TestLeaderboard:
    def _board(self):
        b = Leaderboard(["MMLU", "GSM8K"])
        b.add_model("A", {"MMLU": 0.9, "GSM8K": 0.8})
        b.add_model("B", {"MMLU": 0.7, "GSM8K": 0.6})
        return b

    def test_ranking_order(self):
        b = self._board()
        ranked = b.ranking()
        assert ranked[0].model_name == "A"   # higher mean score

    def test_best_model(self):
        b = self._board()
        assert b.best_model().model_name == "A"

    def test_elo_update_winner(self):
        b = self._board()
        old_a = b.ranking()[0].elo
        b.update_elo("A", "B")
        new_a = b.ranking()[0].elo
        assert new_a > old_a   # winner's Elo increases

    def test_leaderboard_table(self):
        b = self._board()
        t = b.leaderboard_table()
        assert t[0]["rank"] == 1
        assert "mean_score" in t[0]

    def test_task_comparison(self):
        b = self._board()
        r = b.task_comparison("MMLU")
        assert r[0][1] >= r[1][1]   # sorted descending


class TestExactMatchEdge:
    def test_empty_strings(self):
        assert exact_match("", "") is True

    def test_case_insensitive(self):
        assert exact_match("HELLO", "hello") is True

    def test_punctuation_stripped(self):
        assert exact_match("Paris!", "paris") is True


class TestTokenF1Edge:
    def test_both_empty(self):
        assert token_f1("", "") == 1.0

    def test_one_empty(self):
        assert token_f1("hello", "") == 0.0


class TestMathExactMatchText:
    def test_fraction(self):
        from nanomind.eval import math_exact_match, extract_number
        assert extract_number("4/5") == "4/5"

    def test_negative_number(self):
        from nanomind.eval import math_exact_match
        assert math_exact_match("answer is -3", "-3")


class TestJudgeSwap:
    def test_no_swap(self):
        j    = LLMJudge(swap_debiasing=False)
        pair = j.compare("Q?", "A good answer", "A mediocre answer")
        assert pair.winner in ("A", "B", "tie")

    def test_with_swap(self):
        j    = LLMJudge(swap_debiasing=True)
        pair = j.compare("Q?", "Response A", "Response B")
        assert pair.winner in ("A", "B", "tie")


class TestBenchmarkReport:
    def test_overall_accuracy(self):
        from nanomind.eval import BenchmarkEvaluator
        ev   = BenchmarkEvaluator(lambda s: s.answer, "oracle")
        rep  = ev.evaluate_all([make_mmlu_task(4), make_gsm8k_task(4)])
        assert 0.0 <= rep.overall_accuracy <= 1.0

    def test_total_samples(self):
        from nanomind.eval import BenchmarkEvaluator
        ev  = BenchmarkEvaluator(lambda s: s.answer, "oracle")
        rep = ev.evaluate_all([make_mmlu_task(4), make_gsm8k_task(4)])
        assert rep.total_samples == 8
