"""
examples/diffusion_demo.py — NanoMind Diffusion Language Model demo.

Demonstrates:
  1. NoiseSchedule: linear, cosine, sigmoid — q_sample, SNR
  2. DiffusionDenoiser: AdaLN transformer, sinusoidal time embedding
  3. DDPM: training loss, ancestral sampling
  4. DDIM: 10× faster sampling
  5. MaskedDiffusionLM: D3PM-style token masking
  6. ClassifierFreeGuidance: CFG noise prediction
  7. DiffusionLMPipeline: end-to-end training + generation

Usage:
    python examples/diffusion_demo.py
"""
import torch
from nanomind.diffusion import (
    NoiseSchedule, DiffusionDenoiser, SinusoidalTimeEmbedding,
    DDPMTrainer, DDPMSampler, DDIMSampler,
    MaskedDiffusionLM, ClassifierFreeGuidance,
    DiffusionLMPipeline,
)

V   = 32
print("=" * 60)
print("NanoMind Diffusion Language Model Demo")
print("=" * 60)

# ── NoiseSchedule ─────────────────────────────────────────────────────────────
print("
── Noise Schedules ──")
for sched in ["linear", "cosine", "sigmoid"]:
    ns = NoiseSchedule(n_steps=100, schedule=sched)
    print(f"  {sched:8}: {ns.to_dict()}")

ns = NoiseSchedule(n_steps=100, schedule="cosine")
x0 = torch.randn(2, 8, 16)     # (B, T, D) continuous embeddings
t  = torch.tensor([10, 50])
x_t, noise = ns.q_sample(x0, t)
print(f"
  q_sample: x_t.shape={tuple(x_t.shape)}, noise.shape={tuple(noise.shape)}")
print(f"  SNR at t=0: {ns.snr(0):.2f}, t=50: {ns.snr(50):.4f}, t=99: {ns.snr(99):.6f}")

# ── SinusoidalTimeEmbedding ───────────────────────────────────────────────────
print("
── Sinusoidal Time Embedding ──")
te = SinusoidalTimeEmbedding(dim=32)
t_emb = te(torch.tensor([0, 50, 99]))
print(f"  t_emb shape: {tuple(t_emb.shape)}")

# ── DiffusionDenoiser ─────────────────────────────────────────────────────────
print("
── DiffusionDenoiser (AdaLN Transformer) ──")
denoiser = DiffusionDenoiser(vocab_size=V, d_model=32, n_layers=2, n_heads=2, max_seq=16)
print(f"  Params: {denoiser.n_params:,}")
x_noisy = torch.randn(2, 8, 32)
t_ids   = torch.randint(0, 100, (2,))
with torch.no_grad():
    eps_pred = denoiser(x_noisy, t_ids)
print(f"  eps_pred shape: {tuple(eps_pred.shape)}")
tok_ids = denoiser.decode_to_tokens(x_noisy)
print(f"  decode_to_tokens: {tuple(tok_ids.shape)}")

# ── DDPM Training ─────────────────────────────────────────────────────────────
print("
── DDPM Training ──")
trainer = DDPMTrainer(denoiser, ns)
ids     = torch.randint(0, V, (2, 8))
loss_simple = trainer.loss(ids, "simple")
loss_vlb    = trainer.loss(ids, "vlb")
print(f"  Simple loss: {loss_simple.item():.4f}")
print(f"  VLB loss:    {loss_vlb.item():.4f}")
print(f"  Loss at t=0: {trainer.loss_at_t(ids, 0):.4f}")
print(f"  Loss at t=99: {trainer.loss_at_t(ids, 99):.4f}")

# ── DDPM Sampling ─────────────────────────────────────────────────────────────
print("
── DDPM Sampling (10 steps) ──")
ddpm_sampler = DDPMSampler(denoiser, ns)
samples = ddpm_sampler.sample(batch_size=2, seq_len=8, n_steps=10)
print(f"  Generated shape: {tuple(samples.shape)}")

# ── DDIM Sampling ─────────────────────────────────────────────────────────────
print("
── DDIM Sampling (5 steps — fast!) ──")
ddim_sampler = DDIMSampler(denoiser, ns, eta=0.0)
samples_ddim = ddim_sampler.sample(batch_size=2, seq_len=8, n_steps=5)
print(f"  DDIM generated shape: {tuple(samples_ddim.shape)}")

# ── Masked Diffusion ──────────────────────────────────────────────────────────
print("
── Masked Diffusion LM (D3PM style) ──")
mdlm   = MaskedDiffusionLM(vocab_size=V, d_model=32, n_layers=2)
ids    = torch.randint(2, V, (2, 8))   # avoid MASK_ID=1
masked, mask = mdlm.forward_mask(ids, t=50, T=100)
print(f"  Mask rate at t=50: {mask.float().mean():.2f}")
loss_m = mdlm.loss(ids, t=50, T=100)
print(f"  Masked diffusion loss: {loss_m.item():.4f}")
gen = mdlm.generate(batch_size=2, seq_len=8, T=5)
print(f"  Generated tokens: {tuple(gen.shape)}")

# ── Classifier-Free Guidance ──────────────────────────────────────────────────
print("
── Classifier-Free Guidance ──")
cfg     = ClassifierFreeGuidance(denoiser, guidance_scale=7.5)
x_t_cfg = torch.randn(2, 8, 32)
t_cfg   = torch.randint(0, 100, (2,))
cond    = torch.randn(2, 32)
eps_cfg = cfg.guided_predict(x_t_cfg, t_cfg, cond)
print(f"  CFG eps shape: {tuple(eps_cfg.shape)}")
print(f"  Dynamic scale at t=50: {cfg.guidance_scale_schedule(50, 100):.2f}")

# ── DiffusionLMPipeline ───────────────────────────────────────────────────────
print("
── DiffusionLMPipeline (continuous) ──")
pipe = DiffusionLMPipeline(vocab_size=V, d_model=32, n_layers=2, n_steps=50)
for _ in range(3):
    loss = pipe.train_step(torch.randint(0, V, (2, 8)))
print(f"  Training loss: {loss:.4f}")
tokens = pipe.generate(batch_size=2, seq_len=8, n_steps=5, sampler="ddim")
print(f"  Generated: {tuple(tokens.shape)}")
print(f"  Pipeline info: {pipe.info()}")

print("
── DiffusionLMPipeline (masked) ──")
pipe_m = DiffusionLMPipeline(vocab_size=V, d_model=32, n_layers=2, mode="masked")
for _ in range(3):
    loss_m = pipe_m.train_step(torch.randint(2, V, (2, 8)))
print(f"  Masked training loss: {loss_m:.4f}")
tokens_m = pipe_m.generate(batch_size=2, seq_len=8, n_steps=5)
print(f"  Masked generated: {tuple(tokens_m.shape)}")
print("
Diffusion LM demo complete!")
