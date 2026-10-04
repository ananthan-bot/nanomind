"""
examples/eval_demo.py — NanoMind Evaluation & Benchmarking demo.

Usage:
    python examples/eval_demo.py
"""
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

V = 32

class TinyLM(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.emb  = nn.Embedding(V, d)
        self.rnn  = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x % V))
        return self.head(h), None

print("=" * 60)
print("NanoMind Evaluation & Benchmarking Demo")
print("=" * 60)

# ── Benchmarks ────────────────────────────────────────────────────────────────
print("
── Benchmark Tasks ──")
mmlu     = make_mmlu_task(n=10)
gsm8k    = make_gsm8k_task(n=8)
hella    = make_hellaswag_task(n=6)
truthful = make_truthfulqa_task(n=6)

for task in [mmlu, gsm8k, hella, truthful]:
    print(f"  {task.name}: {len(task)} samples, "
          f"type={task.task_type.value}, "
          f"subjects={task.subjects()[:2]}")

# ── EvalSample ────────────────────────────────────────────────────────────────
print("
── EvalSample ──")
s = mmlu.samples[0]
print(f"  ID: {s.sample_id}")
print(f"  Q:  {s.question}")
print(f"  Choices:
{s.format_choices()}")
print(f"  Answer: {s.answer} → '{s.answer_text}'")

# ── Metrics ───────────────────────────────────────────────────────────────────
print("
── Metrics ──")
print(f"  exact_match('Paris', 'Paris'):   {exact_match('Paris', 'Paris')}")
print(f"  exact_match('Paris', 'London'):  {exact_match('Paris', 'London')}")
print(f"  token_f1('the cat sat', 'cat sat mat'): {token_f1('the cat sat', 'cat sat mat'):.3f}")
print(f"  rouge_l('the cat', 'the cat sat'):      {rouge_l('the cat', 'the cat sat'):.3f}")
print(f"  math_exact_match('The answer is 42', '42'): {math_exact_match('The answer is 42', '42')}")

preds = ["A", "B", "A", "C", "A"]
refs  = ["A", "B", "C", "C", "B"]
acc   = multiple_choice_accuracy(preds, refs)
print(f"  MC accuracy: {acc.value:.2%} ({acc.to_dict()})")

gen_m = generation_metrics(["the cat sat on the mat"],
                             ["the cat sat on the mat"])
print(f"  Generation metrics: {gen_m}")

# Perplexity
model = TinyLM()
ids   = torch.randint(0, V, (1, 32))
ppl   = perplexity(model, ids)
print(f"  Perplexity (random model): {ppl:.2f}")

# ── LLM-as-a-Judge ────────────────────────────────────────────────────────────
print("
── LLM-as-a-Judge ──")
judge = LLMJudge(swap_debiasing=True)

result = judge.score_response(
    "What is the capital of France?",
    "The capital of France is Paris.",
)
print(f"  Score: {result.score}/10, Normalized: {result.normalized_score:.2%}")

pair = judge.compare(
    "Explain what DNA is.",
    "DNA is deoxyribonucleic acid, containing genetic information.",
    "I don't know.",
)
print(f"  Pairwise: Winner={pair.winner}, A={pair.score_a}/10, B={pair.score_b}/10")

win_rates = judge.win_rate(
    ["Q1", "Q2", "Q3"],
    ["Good A1", "Good A2", "Good A3"],
    ["Bad B1", "Bad B2", "Bad B3"],
)
print(f"  Win rates: {win_rates}")

# ── Benchmark Evaluator ───────────────────────────────────────────────────────
print("
── Benchmark Evaluator ──")

# Mock model: always answers "A" for MC, "42" for math
def mock_model_fn(sample: EvalSample) -> str:
    if sample.task_type == TaskType.MULTIPLE_CHOICE:
        return "A"
    if sample.task_type == TaskType.MATH_REASONING:
        return sample.answer   # perfect oracle for demo
    return "Paris"

evaluator = BenchmarkEvaluator(mock_model_fn, model_name="MockModel")
for task in [mmlu, gsm8k, hella]:
    result = evaluator.evaluate_task(task)
    print(f"  {task.name}: acc={result.accuracy:.2%}, "
          f"by_subject={list(result.by_subject.items())[:2]}")

report = evaluator.evaluate_all([mmlu, gsm8k, hella])
print(f"
  Overall report:")
print(f"    Model: {report.model_name}")
print(f"    Overall accuracy: {report.overall_accuracy:.2%}")
print(f"    Total samples: {report.total_samples}")
for t in report.tasks:
    print(f"    {t.task_name}: {t.accuracy:.2%}")

# ── Leaderboard ───────────────────────────────────────────────────────────────
print("
── Model Leaderboard ──")
board = Leaderboard(["MMLU", "GSM8K", "HellaSwag", "TruthfulQA"])
board.add_model("GPT-4",       {"MMLU": 0.86, "GSM8K": 0.92, "HellaSwag": 0.96, "TruthfulQA": 0.71})
board.add_model("LLaMA-70B",   {"MMLU": 0.79, "GSM8K": 0.77, "HellaSwag": 0.87, "TruthfulQA": 0.52})
board.add_model("Mistral-7B",  {"MMLU": 0.63, "GSM8K": 0.52, "HellaSwag": 0.81, "TruthfulQA": 0.44})
board.add_model("NanoMind-v5", {"MMLU": 0.35, "GSM8K": 0.21, "HellaSwag": 0.52, "TruthfulQA": 0.38})

for entry in board.leaderboard_table():
    print(f"  #{entry['rank']}: {entry['model']:15s} "
          f"avg={entry['mean_score']:.2%} "
          f"Elo={entry['elo']:.0f}")

# Elo update from pairwise comparison
board.update_elo("GPT-4", "LLaMA-70B", k=32)
print(f"
  After GPT-4 beats LLaMA-70B:")
best = board.best_model()
print(f"  Best model: {best.model_name} (Elo={best.elo:.1f})")

print(f"
  Task comparison MMLU:")
for name, score in board.task_comparison("MMLU"):
    print(f"    {name}: {score:.2%}")

print("
Evaluation demo complete!")
