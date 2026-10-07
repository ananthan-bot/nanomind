"""
examples/reasoning_demo.py — End-to-end Demonstration of Reasoning & Test-Time Compute.
Demonstrates:
  1. Process Reward Model (PRM) step scoring
  2. Monte Carlo Tree Search (MCTS) exploration of reasoning paths
  3. Group Relative Policy Optimization (GRPO) advantage calculation
  4. Test-Time Compute Scaling Law simulation (Pass@k vs FLOPs)
"""
import torch
from nanomind.reasoning import (
    ReasoningConfig,
    ProcessRewardModel,
    aggregate_step_scores,
    MonteCarloTreeSearch,
    SearchConfig,
    StepBeamSearch,
    BestOfNVerifier,
    SelfReflectiveReasoner,
    compute_group_advantages,
    grpo_loss,
    pass_at_k,
    TestTimeScalingSimulator,
)

def run_demo():
    print("=" * 70)
    print("  NanoMind Day 57: Reasoning Models & Test-Time Compute (o1 / R1)")
    print("=" * 70)

    # 1. Process Reward Model (PRM)
    print("\n[1] Process Reward Model (PRM) Step Verification:")
    prm = ProcessRewardModel(d_model=64)
    # Simulated 3 steps: step 1 correct (0.95), step 2 correct (0.90), step 3 mistake (0.20)
    sim_step_scores = [0.95, 0.90, 0.20]
    p_prod = aggregate_step_scores(sim_step_scores, "product")
    p_min = aggregate_step_scores(sim_step_scores, "min")
    p_last = aggregate_step_scores(sim_step_scores, "last")
    print(f"  Step probabilities: {sim_step_scores}")
    print(f"  PRM product aggregation: {p_prod:.4f} (detects failure along chain)")
    print(f"  PRM min aggregation (weakest link): {p_min:.4f}")
    print(f"  PRM last aggregation: {p_last:.4f}")

    # 2. Monte Carlo Tree Search (MCTS)
    print("\n[2] Monte Carlo Tree Search (MCTS) over Math Reasoning Steps:")
    cfg = SearchConfig(num_simulations=20, c_puct=1.414)
    mcts = MonteCarloTreeSearch(cfg)

    # Simple step candidate generator
    def step_gen(state):
        if "Step 2" in state:
            return [("Step 3: Final Answer is \\boxed{42}", 0.9, True)]
        elif "Step 1" in state:
            return [
                ("Step 2 (branch A): Multiply by 2", 0.7, False),
                ("Step 2 (branch B): Divide by 2", 0.3, False),
            ]
        else:
            return [("Step 1: Simplify expression", 1.0, False)]

    def verifier(prefix, step):
        if "Multiply by 2" in step or "boxed{42}" in step or "Step 1" in step:
            return 0.95
        return 0.15

    trajectory, root = mcts.search("Problem: Solve 2x + 10 = 94", step_gen, verifier)
    print("  MCTS Discovered Optimal Trajectory:")
    for i, s in enumerate(trajectory, 1):
        print(f"    {i}. {s}")

    # 3. Group Relative Policy Optimization (GRPO)
    print("\n[3] Group Relative Policy Optimization (GRPO) Advantage Normalization:")
    rewards = torch.tensor([[1.0, 0.0, 0.5, 1.0]])  # Group of G=4 outputs
    adv = compute_group_advantages(rewards)
    print(f"  Group Rewards: {rewards.tolist()[0]}")
    adv_list = adv.tolist() if adv.dim() == 1 else adv.tolist()[0]
    print(f"  Group Relative Advantages (mean=0, std=1): {[round(x, 3) for x in adv_list]}")

    # 4. Test-Time Scaling Laws (Pass@k)
    print("\n[4] Test-Time Compute Scaling Laws:")
    sim = TestTimeScalingSimulator(p_step_correct=0.88, num_steps=3)
    k_vals = [1, 2, 4, 8, 16]
    tradeoffs = sim.compute_scaling_tradeoff(k_vals)
    print("  Rollouts | Accuracy | Estimated GFLOPs")
    print("  ---------+----------+-----------------")
    for t in tradeoffs:
        print(f"     {t['budget_rollouts']:2d}    |  {t['accuracy']*100:5.1f}%  |  {t['estimated_gflops']:6.2f} GFLOPs")

    print("\n[OK] All Reasoning demos completed successfully!")

if __name__ == "__main__":
    run_demo()
