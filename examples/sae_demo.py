"""
examples/sae_demo.py — End-to-end demonstration of Sparse Autoencoders (Top-K, JumpReLU, Gated),
dead neuron tracking, concept steering, and benchmark evaluation in NanoMind.
"""
import torch
from nanomind.sae import (
    SAEConfig,
    SAEType,
    TopKSAE,
    JumpReLUSAE,
    GatedSAE,
    SAETrainer,
    FeatureSteerer,
    SAEBenchmark,
)


def run_demo():
    print("======================================================================")
    print("  NanoMind Day 66: Sparse Autoencoders & Monosemantic Dictionary Learning")
    print("======================================================================\n")

    torch.manual_seed(42)
    d_in = 64
    d_sae = 256
    batch_size = 32

    # Synthesize multi-concept activations
    concepts = torch.randn(d_sae, d_in)
    concept_acts = (torch.rand(batch_size, d_sae) > 0.95).float()
    activations = torch.matmul(concept_acts, concepts) + 0.05 * torch.randn(batch_size, d_in)

    # [1] Top-K SAE Evaluation
    print("[1] Top-K SAE (Gao et al., OpenAI 2024):")
    cfg_topk = SAEConfig(d_in=d_in, d_sae=d_sae, k_sparse=8, sae_type=SAEType.TOPK)
    sae_topk = TopKSAE(cfg_topk)
    out_topk = sae_topk(activations)
    print(f"  Exact Target Sparsity K: {cfg_topk.k_sparse}")
    print(f"  Measured L0 Sparsity   : {out_topk.l0}")
    print(f"  Initial MSE Loss       : {out_topk.mse:.4f}\n")

    # [2] JumpReLU SAE Evaluation
    print("[2] JumpReLU SAE (Rajamanoharan et al., DeepMind 2024):")
    cfg_jump = SAEConfig(d_in=d_in, d_sae=d_sae, threshold=0.15, sae_type=SAEType.JUMPRELU)
    sae_jump = JumpReLUSAE(cfg_jump)
    out_jump = sae_jump(activations)
    print(f"  Threshold Parameter    : {cfg_jump.threshold}")
    print(f"  Measured L0 Sparsity   : {out_jump.l0:.2f}")
    print(f"  Initial MSE Loss       : {out_jump.mse:.4f}\n")

    # [3] Gated SAE Evaluation
    print("[3] Gated SAE (Dual-path decoupling):")
    cfg_gated = SAEConfig(d_in=d_in, d_sae=d_sae, l1_coeff=1e-3, sae_type=SAEType.GATED)
    sae_gated = GatedSAE(cfg_gated)
    out_gated = sae_gated(activations)
    print(f"  Decoupled Magnitude L1 : {out_gated.l1:.4f}")
    print(f"  Measured L0 Sparsity   : {out_gated.l0:.2f}\n")

    # [4] Fast Dictionary Training Simulation
    print("[4] Fast Dictionary Optimization (10 steps):")
    trainer = SAETrainer(sae_topk, cfg_topk)
    for step in range(10):
        step_out = trainer.train_step(activations)
    print(f"  Trained Top-K MSE      : {step_out.mse:.4f}")
    print(f"  Dead Feature Ratio     : {trainer.tracker.dead_ratio * 100:.1f}%\n")

    # [5] Feature Steering (Golden Gate Concept Clamping)
    print("[5] Concept Intervention & Steering:")
    steerer = FeatureSteerer(sae_topk)
    feat_id = 42
    steered = steerer.steer_activation(activations, feature_idx=feat_id, coefficient=3.0)
    drift = (steered - activations).norm(dim=-1).mean().item()
    print(f"  Target Concept Atom    : Feature #{feat_id}")
    print(f"  Steering Coefficient   : +3.0")
    print(f"  Activation Shift Norm  : {drift:.4f}\n")

    # [6] Full Dictionary Benchmark
    print("[6] SAE Benchmark Suite:")
    bench = SAEBenchmark.evaluate(sae_topk, activations)
    print(f"  NMSE                   : {bench['nmse']}")
    print(f"  Explained Variance     : {bench['explained_variance']}")
    print(f"  L0 Sparsity            : {bench['l0_sparsity']}")
    print(f"  Dead Feature Ratio     : {bench['dead_feature_ratio']}")

    print("\n[OK] NanoMind Sparse Autoencoders demo completed successfully!")


if __name__ == "__main__":
    run_demo()
