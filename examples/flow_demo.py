"""
examples/flow_demo.py — End-to-end Demonstration of Discrete Diffusion & Flow Matching LLMs.
Demonstrates:
  1. Forward Absorbing State Noise Schedule (Cosine)
  2. FlowTransformer Non-Causal Bidirectional Denoising
  3. Parallel Non-Autoregressive Sampling in K=10 Steps
  4. Optimal Transport Flow Matching (OT-CFM) Velocity Field
  5. Autoregressive vs Diffusion Complexity & Speedup Comparison
"""
import torch
from nanomind.flow import (
    FlowConfig,
    CosineNoiseSchedule,
    FlowTransformer,
    DiscreteDiffusionSampler,
    OptimalTransportFlowMatching,
    FlowODEIntegrator,
    compare_generation_complexity,
)


def run_demo():
    print("=" * 70)
    print("  NanoMind Day 62: Discrete Diffusion & Flow Matching LLMs")
    print("=" * 70)

    # 1. Forward Absorbing State Diffusion Schedule
    print("\n[1] Forward Absorbing-State Noise Schedule:")
    schedule = CosineNoiseSchedule(num_timesteps=10)
    clean_tokens = torch.tensor([[101, 45, 230, 89, 12, 90]])  # 6 tokens
    t = torch.tensor([5])  # Halfway through diffusion
    corrupted = schedule.sample_corrupted_tokens(clean_tokens, t, mask_token_id=0)
    print(f"  Clean Tokens (t=0)     : {clean_tokens.tolist()[0]}")
    print(f"  Corrupted Tokens (t=5) : {corrupted.tolist()[0]} (0 is [MASK])")

    # 2. Bidirectional FlowTransformer Forward Pass
    print("\n[2] FlowTransformer Bidirectional Forward Pass:")
    cfg = FlowConfig(vocab_size=500, d_model=64, n_layers=2, n_heads=2, diffusion_steps=10)
    model = FlowTransformer(cfg)
    t_batch = torch.tensor([5])
    logits = model(corrupted, t_batch)
    print(f"  Input tokens shape    : {corrupted.shape}")
    print(f"  Output logits shape   : {logits.shape} (full sequence scored in parallel)")

    # 3. Parallel Non-Autoregressive Generation
    print("\n[3] Parallel Non-Autoregressive Generation (K=10 steps):")
    sampler = DiscreteDiffusionSampler(model, cfg)
    generated = sampler.sample(batch_size=1, seq_len=8)
    print(f"  Generated 8-token sequence in 10 parallel steps: {generated.tolist()[0]}")

    # 4. Optimal Transport Continuous Flow Matching
    print("\n[4] Optimal Transport Continuous Flow Matching (OT-CFM):")
    cfm = OptimalTransportFlowMatching(sigma=1e-4)
    x_noise = torch.randn(1, 4, 64)
    x_target = torch.randn(1, 4, 64)
    t_cfm = torch.tensor([[[0.5]]])
    x_t, u_t = cfm.sample_location_and_target(x_noise, x_target, t_cfm)
    print(f"  Interpolated Point x_t shape : {x_t.shape}")
    print(f"  Target Velocity u_t shape    : {u_t.shape}")

    # 5. Speedup & Complexity Comparison
    print("\n[5] Non-Autoregressive Speedup Analysis:")
    cmp = compare_generation_complexity(seq_len=256, diffusion_steps=16)
    print(f"  Target Sequence Length       : {cmp['sequence_length']} tokens")
    print(f"  Autoregressive Forward Passes: {cmp['autoregressive_forward_passes']}")
    print(f"  Diffusion Forward Passes     : {cmp['diffusion_forward_passes']}")
    print(f"  Theoretical Parallel Speedup : {cmp['theoretical_speedup_factor']}x faster!")

    print("\n[OK] NanoMind Discrete Diffusion & Flow Matching demo completed successfully!")


if __name__ == "__main__":
    run_demo()
