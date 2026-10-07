"""
nanomind/reasoning/scaling.py — Test-Time Compute Scaling laws, FLOP estimator, and Pass@k analysis.
"""
import math
from typing import List, Dict, Any, Tuple, Optional


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of Pass@k (Chen et al. 2021).
    Probability that at least 1 out of k randomly selected samples is correct,
    given n total generated samples with c correct ones.
    Formula: 1 - prod_{i=0}^{k-1} (n - c - i) / (n - i)
    """
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0

    res = 1.0
    for i in range(k):
        res *= (n - c - i) / (n - i)
    return float(1.0 - res)


def estimate_test_time_flops(
    num_tokens: int,
    num_rollouts: int,
    model_params: int = 125_000_000,
) -> float:
    """
    Estimate test-time compute in Floating Point Operations (FLOPs).
    Standard Transformer forward pass ~ 2 * P FLOPs per token.
    Total FLOPs = 2 * P * num_tokens * num_rollouts.
    """
    return float(2.0 * model_params * num_tokens * num_rollouts)


class TestTimeScalingSimulator:
    """
    Simulates test-time scaling curves: Compute Budget (FLOPs) vs Task Accuracy.
    Compares Greedy (k=1), Best-of-N, Majority Voting, and Tree Search.
    """

    def __init__(self, p_step_correct: float = 0.85, num_steps: int = 4):
        self.p_step_correct = p_step_correct
        self.num_steps = num_steps

    def simulate_pass_at_k(self, k_values: List[int], n_total: int = 64) -> Dict[int, float]:
        """
        Simulate pass@k for an end-to-end task with compounding step failure.
        p_full = (p_step)^num_steps
        """
        p_full = self.p_step_correct ** self.num_steps
        c_expected = int(round(p_full * n_total))
        c_expected = max(1, min(n_total, c_expected))

        return {k: pass_at_k(n_total, c_expected, k) for k in k_values if k <= n_total}

    def compute_scaling_tradeoff(self, budgets: List[int]) -> List[Dict[str, Any]]:
        """
        Compute estimated pass rates and FLOPs for varying rollout budgets.
        budgets: list of rollout counts [1, 2, 4, 8, 16, 32]
        """
        results = []
        p_full = self.p_step_correct ** self.num_steps

        for b in budgets:
            # Probability of at least one correct path in b rollouts = 1 - (1 - p_full)^b
            acc = 1.0 - ((1.0 - p_full) ** b)
            flops = estimate_test_time_flops(num_tokens=128 * self.num_steps, num_rollouts=b)
            results.append({
                "budget_rollouts": b,
                "accuracy": float(acc),
                "estimated_gflops": flops / 1e9,
            })
        return results
