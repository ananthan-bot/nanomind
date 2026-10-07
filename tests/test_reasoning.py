"""
tests/test_reasoning.py — Comprehensive unit tests for Reasoning & Test-Time Compute.
"""
import math
import pytest
import torch
from nanomind.reasoning import (
    PRMConfig,
    StepDelimiter,
    ProcessRewardModel,
    aggregate_step_scores,
    PRMLoss,
    OutcomeRewardModel,
    MarginRankingLoss,
    compute_brier_score,
    compute_calibration_error,
    compare_prm_vs_orm,
    ReasoningNode,
    puct_score,
    MonteCarloTreeSearch,
    TreeOfThoughts,
    StepBeamSearch,
    BestOfNVerifier,
    SelfReflectiveReasoner,
    compute_group_advantages,
    grpo_loss,
    GRPOTrainer,
    STaR,
    pass_at_k,
    estimate_test_time_flops,
    TestTimeScalingSimulator,
)


class TestPRM:
    def test_step_delimiter(self):
        delim = StepDelimiter(delimiter="\n\n")
        text = "Step 1: start\n\nStep 2: middle\n\nStep 3: end"
        steps = delim.split(text)
        assert len(steps) == 3
        assert steps[0] == "Step 1: start"
        rejoined = delim.join(steps)
        assert rejoined == text

    def test_prm_forward(self):
        prm = ProcessRewardModel(d_model=32, num_classes=2)
        h = torch.randn(2, 10, 32)
        step_indices = [3, 7, 9]
        probs = prm(h, step_indices=step_indices)
        assert probs.shape == (2, 3)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_aggregate_scores(self):
        scores = [0.9, 0.8, 0.5]
        prod = aggregate_step_scores(scores, "product")
        assert math.isclose(prod, 0.9 * 0.8 * 0.5, rel_tol=1e-5)

        min_val = aggregate_step_scores(scores, "min")
        assert min_val == 0.5

        mean_val = aggregate_step_scores(scores, "mean")
        assert math.isclose(mean_val, (0.9 + 0.8 + 0.5) / 3, rel_tol=1e-5)

    def test_prm_loss(self):
        loss_fn = PRMLoss()
        probs = torch.tensor([[0.8, 0.2]])
        targets = torch.tensor([[1, 0]])
        loss = loss_fn(probs, targets)
        assert loss.item() > 0.0


class TestORM:
    def test_orm_forward(self):
        orm = OutcomeRewardModel(d_model=32)
        h = torch.randn(4, 32)
        scores = orm(h)
        assert scores.shape == (4,)

    def test_margin_ranking_loss(self):
        loss_fn = MarginRankingLoss(margin=0.5)
        chosen = torch.tensor([2.0, 1.5])
        rejected = torch.tensor([0.5, 0.0])
        loss = loss_fn(chosen, rejected)
        assert loss.item() > 0.0

    def test_brier_and_calibration(self):
        preds = [0.9, 0.8, 0.1, 0.2]
        targets = [1, 1, 0, 0]
        brier = compute_brier_score(preds, targets)
        assert brier < 0.05  # Highly calibrated and accurate

        ece = compute_calibration_error(preds, targets, n_bins=5)
        assert 0.0 <= ece <= 1.0


class TestMCTS:
    def test_reasoning_node_and_puct(self):
        root = ReasoningNode(state_text="Question")
        child1 = root.add_child("Step 1", prior_p=0.8)
        child2 = root.add_child("Step 2", prior_p=0.2)

        assert not root.is_leaf()
        assert child1.is_leaf()
        assert child1.visits == 0

        score1 = puct_score(child1, parent_visits=4)
        score2 = puct_score(child2, parent_visits=4)
        assert score1 > score2  # Higher prior gives higher initial PUCT

    def test_mcts_search_convergence(self):
        mcts = MonteCarloTreeSearch()

        def mock_step_gen(state):
            if "Target" in state:
                return [("Done", 1.0, True)]
            return [("Target Step", 0.8, False), ("Wrong Step", 0.2, False)]

        def mock_verifier(prefix, step):
            return 0.95 if "Target" in step or "Done" in step else 0.05

        traj, root = mcts.search("Initial Prompt", mock_step_gen, mock_verifier, num_simulations=15)
        assert len(traj) > 0
        assert "Target Step" in traj[0]

    def test_tree_of_thoughts(self):
        tot = TreeOfThoughts(max_depth=3, beam_width=2)
        def gen(path, k):
            return [f"thought_{i}" for i in range(k)]
        def evaluate(path, thought):
            return ("sure", 0.9) if "0" in thought else ("impossible", 0.1)
        def is_sol(path):
            return "thought_0\n\nthought_0" in path

        solutions = tot.solve("Problem", gen, evaluate, is_sol)
        assert isinstance(solutions, list)


class TestBeamSearchAndBestOfN:
    def test_step_beam_search(self):
        beam = StepBeamSearch(beam_width=2, max_steps=3)
        def prop(text, k):
            return ["step_a", "step_b"]
        def scorer(text, step):
            return 0.9 if step == "step_a" else 0.3
        def is_term(text):
            return text.count("step_a") >= 2

        results = beam.search("Prompt", prop, scorer, is_term)
        assert len(results) > 0
        assert results[0]["cumulative_score"] >= results[-1]["cumulative_score"]

    def test_best_of_n_majority_vote(self):
        verifier = BestOfNVerifier()
        candidates = [
            {"text": "Reasoning... \\boxed{42}", "reward": 0.9},
            {"text": "Reasoning... \\boxed{42}", "reward": 0.8},
            {"text": "Reasoning... \\boxed{100}", "reward": 0.4},
        ]
        winner, conf, dist = verifier.majority_vote(candidates, weighted_by_score=True)
        assert winner == "42"
        assert conf > 0.5
        assert "42" in dist and "100" in dist


class TestGRPOAndScaling:
    def test_group_advantages(self):
        rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        adv = compute_group_advantages(rewards)
        assert math.isclose(adv.mean().item(), 0.0, abs_tol=1e-5)
        assert adv[0, 3] > adv[0, 0]

    def test_grpo_loss(self):
        G, T = 4, 8
        log_probs = torch.randn(G, T)
        old_log_probs = log_probs.clone()
        ref_log_probs = log_probs.clone()
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

        loss, metrics = grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages)
        assert isinstance(loss.item(), float)
        assert "policy_loss" in metrics
        assert "kl_divergence" in metrics

    def test_self_reflection_backtracking(self):
        reasoner = SelfReflectiveReasoner()
        text = "<think>Let me compute 7 * 8 = 54. Wait, that's incorrect. Actually 7 * 8 = 56.</think>\\boxed{56}"
        parsed = reasoner.parse_reasoning_trace(text)
        assert parsed["has_think_tags"]
        assert parsed["backtrack_count"] >= 1
        assert parsed["final_answer"] == "56"

    def test_star_bootstrap(self):
        star = STaR()
        dataset = [{"question": "2+2", "answer": "4"}, {"question": "3+3", "answer": "6"}]
        def gen(q):
            return "4" if "2+2" in q else "wrong"
        def evaluate(pred, gt):
            return pred == gt

        res = star.bootstrap_iteration(dataset, gen, evaluate)
        assert res["direct_accuracy"] == 0.5
        assert res["buffer_size"] == 1

    def test_pass_at_k_and_scaling(self):
        # 10 samples, 2 correct: pass@1 should be 0.2
        p1 = pass_at_k(n=10, c=2, k=1)
        assert math.isclose(p1, 0.2, rel_tol=1e-4)

        # pass@k should increase with k
        p5 = pass_at_k(n=10, c=2, k=5)
        assert p5 > p1

        flops = estimate_test_time_flops(num_tokens=100, num_rollouts=4, model_params=1_000_000)
        assert flops == 2.0 * 1_000_000 * 100 * 4

        sim = TestTimeScalingSimulator(p_step_correct=0.9, num_steps=2)
        tradeoffs = sim.compute_scaling_tradeoff([1, 2, 4])
        assert len(tradeoffs) == 3
        assert tradeoffs[-1]["accuracy"] >= tradeoffs[0]["accuracy"]
