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


class TestSEDD:
    def test_score_entropy_loss(self):
        logits = torch.randn(2, 4, 30, requires_grad=True)
        x_t = torch.tensor([[0, 10, 0, 5], [2, 0, 0, 8]])  # 0 is mask
        x_0 = torch.tensor([[1, 10, 3, 5], [2, 7, 9, 8]])
        loss = score_entropy_loss(logits, x_t, x_0, mask_token_id=0)
        assert loss.item() > 0.0
        loss.backward()
        assert logits.grad is not None


class TestFlowMatching:
    def test_ot_flow_matching(self):
        cfm = OptimalTransportFlowMatching()
        x_0 = torch.randn(2, 4, 16)
        x_1 = torch.randn(2, 4, 16)
        t = torch.tensor([[[0.3]]])
        x_t, u_t = cfm.sample_location_and_target(x_0, x_1, t)
        assert x_t.shape == (2, 4, 16)
        assert u_t.shape == (2, 4, 16)

        pred_v = torch.randn_like(u_t)
        loss = cfm.compute_loss(pred_v, u_t)
        assert loss.item() > 0.0


class TestModelAndSampler:
    def test_flow_transformer_forward(self):
        cfg = FlowConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=2)
        model = FlowTransformer(cfg)
        x_t = torch.randint(0, 100, (2, 6))
        t = torch.tensor([3, 3])
        logits = model(x_t, t)
        assert logits.shape == (2, 6, 100)

    def test_discrete_diffusion_sampler(self):
        cfg = FlowConfig(vocab_size=100, d_model=32, n_layers=1, n_heads=2, diffusion_steps=4)
        model = FlowTransformer(cfg)
        sampler = DiscreteDiffusionSampler(model, cfg)
        out = sampler.sample(batch_size=1, seq_len=4)
        assert out.shape == (1, 4)

    def test_speedup_and_diversity(self):
        cmp = compare_generation_complexity(seq_len=100, diffusion_steps=10)
        assert cmp["theoretical_speedup_factor"] == 10.0

        toks = torch.tensor([[1, 2, 3, 4]])
        div = compute_token_diversity(toks)
        assert div == 1.0
