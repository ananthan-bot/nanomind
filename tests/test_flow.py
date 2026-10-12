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
