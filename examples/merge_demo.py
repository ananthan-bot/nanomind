"""
examples/merge_demo.py — End-to-end Demonstration of Model Merging (TIES, DARE, SLERP, Soups).
Demonstrates:
  1. Task Vector extraction & arithmetic (addition & task negation)
  2. TIES-Merging across two fine-tuned models
  3. DARE (Drop And REscale) extreme parameter pruning
  4. Spherical Linear Interpolation (SLERP)
  5. Parameter drift computation
"""
import copy
import torch
import torch.nn as nn

from nanomind.merge import (
    MergeConfig,
    MergeMethod,
    ModelMerger,
    TaskVector,
    TIESMerger,
    DAREPruner,
    slerp,
    ModelSoup,
    compute_parameter_drift,
    compute_task_vector_cosine_similarity,
)


def run_demo():
    print("=" * 70)
    print("  NanoMind Day 63: Model Merging & Weight Averaging (MergeKit)")
    print("=" * 70)

    # Simple 2-layer MLP as base model
    torch.manual_seed(42)
    base_model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )

    # Simulate two fine-tuned models (e.g. Model A tuned for Math, Model B tuned for Code)
    ft_model_a = copy.deepcopy(base_model)
    ft_model_b = copy.deepcopy(base_model)

    with torch.no_grad():
        for p in ft_model_a.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        for p in ft_model_b.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # 1. Task Vectors & Task Arithmetic
    print("\n[1] Task Vector Computation & Arithmetic:")
    tv_a = TaskVector.from_models(base_model, ft_model_a)
    tv_b = TaskVector.from_models(base_model, ft_model_b)
    tv_sum = tv_a + tv_b
    cos_sim = compute_task_vector_cosine_similarity(tv_a.vector_dict, tv_b.vector_dict)
    print(f"  Task Vector A norm: {sum(v.norm().item() for v in tv_a.vector_dict.values()):.3f}")
    print(f"  Task Vector B norm: {sum(v.norm().item() for v in tv_b.vector_dict.values()):.3f}")
    print(f"  Task Vector Cosine Similarity (Interference): {cos_sim:.4f}")

    # 2. TIES-Merging
    print("\n[2] TIES-Merging (Trim, Elect Sign, Disjoint Merge):")
    cfg_ties = MergeConfig(method=MergeMethod.TIES, density=0.3)
    merger_ties = ModelMerger(cfg_ties)
    merged_ties = merger_ties.merge(base_model, [ft_model_a, ft_model_b])
    drift_ties = compute_parameter_drift(base_model, merged_ties)
    print(f"  TIES Merged Model Drift from Base: {drift_ties:.4f}")

    # 3. DARE (Drop And REscale)
    print("\n[3] DARE Parameter Pruning & Rescaling:")
    dare = DAREPruner(drop_rate=0.5, rescale=True)
    sample_tensor = torch.randn(4, 4)
    pruned = dare.prune_tensor(sample_tensor)
    sparsity = (pruned == 0.0).float().mean().item()
    print(f"  Original Tensor Mean: {sample_tensor.mean().item():.3f}")
    print(f"  DARE Sparsity: {sparsity * 100:.1f}% dropped")
    print(f"  Rescaled Mean (Preserves Expectation): {pruned.mean().item():.3f}")

    # 4. Spherical Linear Interpolation (SLERP)
    print("\n[4] Spherical Linear Interpolation (SLERP):")
    p1 = torch.tensor([1.0, 0.0, 0.0])
    p2 = torch.tensor([0.0, 1.0, 0.0])
    interp = slerp(p1, p2, t=0.5)
    print(f"  P1: {p1.tolist()}, P2: {p2.tolist()}")
    print(f"  SLERP midpoint (t=0.5): {[round(x, 3) for x in interp.tolist()]}")
    print(f"  Midpoint norm: {interp.norm().item():.3f} (preserved on unit sphere)")

    # 5. Model Soup (Uniform Averaging)
    print("\n[5] Model Soup (Uniform Averaging):")
    soup_model = ModelSoup.uniform_soup([ft_model_a, ft_model_b])
    drift_soup = compute_parameter_drift(base_model, soup_model)
    print(f"  Soup Parameter Drift: {drift_soup:.4f}")

    print("\n[OK] NanoMind Model Merging demo completed successfully!")


if __name__ == "__main__":
    run_demo()
