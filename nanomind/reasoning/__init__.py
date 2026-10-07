"""
nanomind.reasoning — Reasoning Models & Test-Time Compute (o1 / DeepSeek-R1 / PRM / MCTS / GRPO).
"""
from nanomind.reasoning.config import (
    ReasoningConfig,
    PRMConfig,
    SearchConfig,
    GRPOConfig,
)
from nanomind.reasoning.prm import (
    StepDelimiter,
    ProcessRewardModel,
    aggregate_step_scores,
    PRMLoss,
)
from nanomind.reasoning.orm import (
    OutcomeRewardModel,
    MarginRankingLoss,
    compute_brier_score,
    compute_calibration_error,
    compare_prm_vs_orm,
)
from nanomind.reasoning.tree_search import (
    ReasoningNode,
    puct_score,
    MonteCarloTreeSearch,
)
from nanomind.reasoning.tot import (
    ThoughtStep,
    TreeOfThoughts,
)
from nanomind.reasoning.beam_search import (
    StepBeamSearch,
    BestOfNVerifier,
)
from nanomind.reasoning.self_correction import (
    SelfReflectiveReasoner,
)
from nanomind.reasoning.grpo import (
    compute_group_advantages,
    grpo_loss,
    GRPOTrainer,
)
from nanomind.reasoning.star import (
    STaR,
)
from nanomind.reasoning.scaling import (
    pass_at_k,
    estimate_test_time_flops,
    TestTimeScalingSimulator,
)

__all__ = [
    "ReasoningConfig",
    "PRMConfig",
    "SearchConfig",
    "GRPOConfig",
    "StepDelimiter",
    "ProcessRewardModel",
    "aggregate_step_scores",
    "PRMLoss",
    "OutcomeRewardModel",
    "MarginRankingLoss",
    "compute_brier_score",
    "compute_calibration_error",
    "compare_prm_vs_orm",
    "ReasoningNode",
    "puct_score",
    "MonteCarloTreeSearch",
    "ThoughtStep",
    "TreeOfThoughts",
    "StepBeamSearch",
    "BestOfNVerifier",
    "SelfReflectiveReasoner",
    "compute_group_advantages",
    "grpo_loss",
    "GRPOTrainer",
    "STaR",
    "pass_at_k",
    "estimate_test_time_flops",
    "TestTimeScalingSimulator",
]
