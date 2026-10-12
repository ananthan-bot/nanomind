"""
tests/test_flow.py — Unit tests for Discrete Diffusion & Flow Matching LLMs.
"""
import torch
from nanomind.flow import (
    FlowConfig,
    CosineNoiseSchedule,
    LinearNoiseSchedule,
    D3PM,
    score_entropy_loss,
    OptimalTransportFlowMatching,
    FlowTransformer,
    DiscreteDiffusionSampler,
    FlowODEIntegrator,
    compare_generation_complexity,
    compute_token_diversity,
)


class TestNoiseSchedules:
    def test_cosine_schedule_bounds(self):
        sched = CosineNoiseSchedule(num_timesteps=10)
        assert sched.alpha_bars[0].item() == 1.0  # Clean at t=0
        assert sched.alpha_bars[-1].item() < 0.1  # Highly corrupted at t=T

    def test_sample_corrupted_tokens(self):
        sched = CosineNoiseSchedule(num_timesteps=10)
        x_0 = torch.tensor([[1, 2, 3, 4]])
        # At t=0, tokens should stay uncorrupted
        x_clean = sched.sample_corrupted_tokens(x_0, torch.tensor([0]), mask_token_id=0)
        assert (x_clean == x_0).all()


class TestD3PM:
    def test_d3pm_sampling_and_posterior(self):
        d3pm = D3PM(vocab_size=50, mask_token_id=0, num_timesteps=10)
        x_0 = torch.randint(1, 50, (2, 8))
        t = torch.tensor([5, 5])
        x_t = d3pm.q_sample(x_0, t)
        assert x_t.shape == (2, 8)

        # Posterior step at t=0 returns argmax
        logits = torch.randn(2, 8, 50)
        x_clean = d3pm.compute_posterior_step(x_t, logits, t=0)
        assert x_clean.shape == (2, 8)
