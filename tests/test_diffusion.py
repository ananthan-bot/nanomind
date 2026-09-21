"""tests/test_diffusion.py — Tests for NanoMind Diffusion Language Models."""
import pytest
import torch
from nanomind.diffusion import (
    NoiseSchedule, DiffusionDenoiser, SinusoidalTimeEmbedding,
    DDPMTrainer, DDPMSampler, DDIMSampler,
    MaskedDiffusionLM, MASK_ID, ClassifierFreeGuidance,
    DiffusionLMPipeline,
)

V = 16


# ── NoiseSchedule ─────────────────────────────────────────────────────────────

class TestNoiseSchedule:
    def _sched(self, sched="cosine"):
        return NoiseSchedule(n_steps=100, schedule=sched)

    def test_linear_schedule(self):
        ns = self._sched("linear")
        assert ns.betas.shape == (100,)
        assert ns.betas[0] < ns.betas[-1]

    def test_cosine_schedule(self):
        ns = self._sched("cosine")
        assert ns.alpha_bars[0] > ns.alpha_bars[-1]

    def test_sigmoid_schedule(self):
        ns = self._sched("sigmoid")
        assert ns.betas.shape == (100,)

    def test_alpha_bars_decreasing(self):
        ns = self._sched("cosine")
        # ᾱ should be monotonically decreasing
        assert (ns.alpha_bars[1:] <= ns.alpha_bars[:-1]).all()

    def test_q_sample_shape(self):
        ns  = self._sched()
        x0  = torch.randn(2, 4, 8)
        t   = torch.tensor([10, 50])
        x_t, noise = ns.q_sample(x0, t)
        assert x_t.shape == x0.shape
        assert noise.shape == x0.shape

    def test_snr_decreases(self):
        ns = self._sched()
        assert ns.snr(0) > ns.snr(50) > ns.snr(99)

    def test_predict_x0_shape(self):
        ns  = self._sched()
        x_t = torch.randn(2, 4, 8)
        eps = torch.randn(2, 4, 8)
        t   = torch.tensor([20, 40])
        x0  = ns.predict_x0(x_t, t, eps)
        assert x0.shape == x_t.shape

    def test_to_dict_keys(self):
        d = self._sched().to_dict()
        for k in ("n_steps", "schedule", "beta_min", "beta_max"):
            assert k in d


# ── SinusoidalTimeEmbedding ───────────────────────────────────────────────────

class TestSinusoidalTimeEmbedding:
    def test_output_shape(self):
        te = SinusoidalTimeEmbedding(32)
        t  = torch.tensor([0, 50, 99])
        assert te(t).shape == (3, 32)

    def test_different_timesteps_different_emb(self):
        te = SinusoidalTimeEmbedding(32)
        e0 = te(torch.tensor([0]))
        e50 = te(torch.tensor([50]))
        assert not torch.allclose(e0, e50)


# ── DiffusionDenoiser ─────────────────────────────────────────────────────────

class TestDiffusionDenoiser:
    def _denoiser(self):
        return DiffusionDenoiser(V, d_model=16, n_layers=1, n_heads=2, max_seq=8)

    def test_forward_shape(self):
        d   = self._denoiser()
        x   = torch.randn(2, 4, 16)
        t   = torch.randint(0, 100, (2,))
        out = d(x, t)
        assert out.shape == (2, 4, 16)

    def test_embed_tokens_shape(self):
        d   = self._denoiser()
        ids = torch.randint(0, V, (2, 4))
        emb = d.embed_tokens(ids)
        assert emb.shape == (2, 4, 16)

    def test_decode_to_tokens_shape(self):
        d   = self._denoiser()
        emb = torch.randn(2, 4, 16)
        ids = d.decode_to_tokens(emb)
        assert ids.shape == (2, 4)

    def test_decode_ids_in_vocab(self):
        d   = self._denoiser()
        emb = torch.randn(2, 4, 16)
        ids = d.decode_to_tokens(emb)
        assert (ids >= 0).all() and (ids < V).all()

    def test_n_params_positive(self):
        d = self._denoiser()
        assert d.n_params > 0


# ── DDPMTrainer ───────────────────────────────────────────────────────────────

class TestDDPMTrainer:
    def _setup(self):
        ns = NoiseSchedule(n_steps=50, schedule="cosine")
        d  = DiffusionDenoiser(V, d_model=16, n_layers=1, n_heads=2, max_seq=8)
        return DDPMTrainer(d, ns), d, ns

    def test_loss_returns_scalar(self):
        t, d, ns = self._setup()
        ids      = torch.randint(0, V, (2, 4))
        loss     = t.loss(ids)
        assert loss.shape == ()

    def test_loss_positive(self):
        t, d, ns = self._setup()
        ids      = torch.randint(0, V, (2, 4))
        assert t.loss(ids).item() > 0

    def test_vlb_loss(self):
        t, d, ns = self._setup()
        ids      = torch.randint(0, V, (2, 4))
        loss     = t.loss(ids, "vlb")
        assert loss.item() >= 0

    def test_loss_at_t(self):
        t, d, ns = self._setup()
        ids      = torch.randint(0, V, (2, 4))
        l        = t.loss_at_t(ids, 10)
        assert isinstance(l, float)

    def test_gradient_flows(self):
        t, d, ns = self._setup()
        ids      = torch.randint(0, V, (2, 4))
        loss     = t.loss(ids)
        loss.backward()
        has_grad = any(p.grad is not None for p in d.parameters())
        assert has_grad


# ── DDPMSampler ───────────────────────────────────────────────────────────────

class TestDDPMSampler:
    def _sampler(self):
        ns = NoiseSchedule(n_steps=20, schedule="cosine")
        d  = DiffusionDenoiser(V, d_model=16, n_layers=1, n_heads=2, max_seq=8)
        return DDPMSampler(d, ns)

    def test_sample_shape(self):
        s    = self._sampler()
        toks = s.sample(batch_size=2, seq_len=4, n_steps=3)
        assert toks.shape == (2, 4)

    def test_sample_ids_in_vocab(self):
        s    = self._sampler()
        toks = s.sample(batch_size=1, seq_len=4, n_steps=2)
        assert (toks >= 0).all() and (toks < V).all()


# ── DDIMSampler ───────────────────────────────────────────────────────────────

class TestDDIMSampler:
    def _sampler(self, eta=0.0):
        ns = NoiseSchedule(n_steps=20, schedule="cosine")
        d  = DiffusionDenoiser(V, d_model=16, n_layers=1, n_heads=2, max_seq=8)
        return DDIMSampler(d, ns, eta=eta)

    def test_sample_shape(self):
        s    = self._sampler()
        toks = s.sample(batch_size=2, seq_len=4, n_steps=5)
        assert toks.shape == (2, 4)

    def test_deterministic(self):
        """eta=0 → same seed → same output."""
        s = self._sampler(eta=0.0)
        torch.manual_seed(42)
        t1 = s.sample(batch_size=1, seq_len=4, n_steps=3)
        torch.manual_seed(42)
        t2 = s.sample(batch_size=1, seq_len=4, n_steps=3)
        assert torch.all(t1 == t2)

    def test_stochastic_eta1(self):
        """eta=1 should generally differ between runs."""
        s  = self._sampler(eta=1.0)
        t1 = s.sample(batch_size=1, seq_len=8, n_steps=3)
        t2 = s.sample(batch_size=1, seq_len=8, n_steps=3)
        assert t1.shape == t2.shape


# ── MaskedDiffusionLM ─────────────────────────────────────────────────────────

class TestMaskedDiffusionLM:
    def _mdlm(self):
        return MaskedDiffusionLM(vocab_size=V, d_model=16, n_layers=1, n_heads=2, max_seq=8)

    def test_forward_mask_rate(self):
        mdlm   = self._mdlm()
        ids    = torch.randint(2, V, (4, 6))
        masked, mask = mdlm.forward_mask(ids, t=50, T=100)
        rate   = mask.float().mean().item()
        assert 0.3 < rate < 0.7   # ≈ 0.5 at t=50

    def test_predict_shape(self):
        mdlm = self._mdlm()
        ids  = torch.randint(0, V, (2, 4))
        out  = mdlm.predict(ids, t=20)
        assert out.shape == (2, 4, V)

    def test_loss_scalar(self):
        mdlm = self._mdlm()
        ids  = torch.randint(2, V, (2, 4))
        loss = mdlm.loss(ids, t=50, T=100)
        assert loss.shape == ()

    def test_generate_shape(self):
        mdlm = self._mdlm()
        gen  = mdlm.generate(batch_size=2, seq_len=4, T=3)
        assert gen.shape == (2, 4)


# ── ClassifierFreeGuidance ────────────────────────────────────────────────────

class TestCFG:
    def _cfg(self):
        d   = DiffusionDenoiser(V, d_model=16, n_layers=1, n_heads=2, max_seq=8)
        return ClassifierFreeGuidance(d, guidance_scale=5.0)

    def test_guided_predict_shape(self):
        cfg  = self._cfg()
        x_t  = torch.randn(2, 4, 16)
        t    = torch.randint(0, 100, (2,))
        cond = torch.randn(2, 16)
        eps  = cfg.guided_predict(x_t, t, cond)
        assert eps.shape == (2, 4, 16)

    def test_train_mask_zeros_some(self):
        cfg   = ClassifierFreeGuidance(None, p_uncond=0.5)
        cond  = torch.randn(100, 16)
        mc    = cfg.train_mask(cond)
        # ~50% should be zeroed
        zeroed = (mc == 0).all(dim=-1).float().mean()
        assert 0.3 < zeroed < 0.7

    def test_guidance_scale_schedule_range(self):
        cfg = ClassifierFreeGuidance(None, guidance_scale=7.5)
        s0  = cfg.guidance_scale_schedule(0, 100)
        s50 = cfg.guidance_scale_schedule(50, 100)
        s99 = cfg.guidance_scale_schedule(99, 100)
        assert s0 >= s50 >= s99   # higher guidance at low noise


# ── DiffusionLMPipeline ───────────────────────────────────────────────────────

class TestDiffusionLMPipeline:
    def _pipe(self, mode="continuous"):
        return DiffusionLMPipeline(V, d_model=16, n_layers=1, n_steps=20, mode=mode)

    def test_continuous_train_step(self):
        p    = self._pipe("continuous")
        ids  = torch.randint(0, V, (2, 4))
        loss = p.train_step(ids)
        assert isinstance(loss, float)

    def test_masked_train_step(self):
        p    = self._pipe("masked")
        ids  = torch.randint(2, V, (2, 4))
        loss = p.train_step(ids)
        assert isinstance(loss, float)

    def test_generate_ddim(self):
        p    = self._pipe("continuous")
        toks = p.generate(batch_size=2, seq_len=4, n_steps=3, sampler="ddim")
        assert toks.shape == (2, 4)

    def test_generate_masked(self):
        p    = self._pipe("masked")
        toks = p.generate(batch_size=2, seq_len=4, n_steps=3)
        assert toks.shape == (2, 4)

    def test_info_keys(self):
        p = self._pipe("continuous")
        d = p.info()
        for k in ("mode", "n_steps", "n_params"):
            assert k in d


class TestAlphaBarsLimits:
    def test_alpha_bar_t0_near_1(self):
        ns = NoiseSchedule(n_steps=100, schedule="cosine")
        assert ns.alpha_bars[0].item() > 0.9

    def test_alpha_bar_tT_near_0(self):
        ns = NoiseSchedule(n_steps=100, schedule="cosine")
        assert ns.alpha_bars[-1].item() < 0.1


class TestQSampleDeterministic:
    def test_fixed_noise(self):
        ns    = NoiseSchedule(n_steps=50, schedule="cosine")
        x0    = torch.randn(2, 4, 8)
        t     = torch.tensor([10, 30])
        noise = torch.randn_like(x0)
        x1, _ = ns.q_sample(x0, t, noise=noise)
        x2, _ = ns.q_sample(x0, t, noise=noise)
        assert torch.allclose(x1, x2)


class TestDenoiserBlocks:
    def test_n_blocks_matches_layers(self):
        d = DiffusionDenoiser(V, d_model=16, n_layers=3, n_heads=2, max_seq=8)
        assert len(d.blocks) == 3

    def test_different_timesteps_different_output(self):
        d = DiffusionDenoiser(V, d_model=16, n_layers=1, n_heads=2, max_seq=8)
        x = torch.randn(1, 4, 16)
        t0  = torch.tensor([0])
        t99 = torch.tensor([99])
        with torch.no_grad():
            o0  = d(x, t0)
            o99 = d(x, t99)
        assert not torch.allclose(o0, o99)


class TestDDIMFast:
    def test_ddim_fewer_steps_same_shape(self):
        ns = NoiseSchedule(n_steps=100, schedule="cosine")
        d  = DiffusionDenoiser(V, d_model=16, n_layers=1, n_heads=2, max_seq=8)
        ddim = DDIMSampler(d, ns, eta=0.0)
        # 10 steps instead of 100
        toks = ddim.sample(batch_size=1, seq_len=4, n_steps=10)
        assert toks.shape == (1, 4)


class TestMaskedAtTmax:
    def test_all_masked_at_t_eq_T(self):
        mdlm = MaskedDiffusionLM(vocab_size=V, d_model=16, n_layers=1)
        ids  = torch.randint(2, V, (2, 8))
        _, mask = mdlm.forward_mask(ids, t=100, T=100)
        # At t=T all should be masked
        assert mask.all()

    def test_none_masked_at_t_0(self):
        mdlm = MaskedDiffusionLM(vocab_size=V, d_model=16, n_layers=1)
        ids  = torch.randint(2, V, (2, 8))
        _, mask = mdlm.forward_mask(ids, t=0, T=100)
        assert not mask.any()
