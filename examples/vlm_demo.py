"""
examples/vlm_demo.py — NanoMind Vision-Language Model demo.

Usage:
    python examples/vlm_demo.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.vlm import (
    VisionEncoderConfig, PatchEmbedding, VisionEncoder,
    CLIPConfig, CLIPLoss, CLIPModel,
    CrossAttentionConfig, PerceiverResampler, VisualProjector, CrossModalAttention,
    VQAConfig, VQASample, EarlyFusionVQA, CrossAttentionVQA, vqa_accuracy,
    CaptioningConfig, ImageCaptioner, bleu_score,
    CrossModalRetriever, RetrievalMetrics,
)

V = 64  # vocab size

class TinyLM(nn.Module):
    def __init__(self, d_in=32, d=64):
        super().__init__()
        self.emb  = nn.Embedding(V, d_in)
        self.rnn  = nn.GRU(d_in, d, batch_first=True)
        self.head = nn.Linear(d, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x % V))
        return self.head(h), None

print("=" * 60)
print("NanoMind Vision-Language Model (VLM) Demo")
print("=" * 60)

# ── Vision Encoder (ViT) ──────────────────────────────────────────────────────
print("
── Vision Encoder (ViT) ──")
cfg     = VisionEncoderConfig(image_size=32, patch_size=8, d_model=32,
                                n_layers=2, n_heads=2, d_ff=64, d_embed=16)
encoder = VisionEncoder(cfg)
imgs    = torch.randn(2, 3, 32, 32)
cls_emb, patch_embs = encoder(imgs)
print(f"  Image: {tuple(imgs.shape)}")
print(f"  Patches: {cfg.n_patches} patches (32/8)² = 16")
print(f"  CLS embedding:    {tuple(cls_emb.shape)}")
print(f"  Patch embeddings: {tuple(patch_embs.shape)}")

# ── Patch Embedding ───────────────────────────────────────────────────────────
print("
── Patch Embedding ──")
pe  = PatchEmbedding(cfg)
out = pe(imgs)
print(f"  PatchEmbedding output: {tuple(out.shape)}  (= 1 CLS + 16 patches)")

# ── CLIP ──────────────────────────────────────────────────────────────────────
print("
── CLIP Contrastive Loss ──")
clip_cfg = CLIPConfig(d_embed=16, temperature_init=0.07)
lm       = TinyLM(d_in=32, d=64)
clip     = CLIPModel(encoder, lm, d_text=64, cfg=clip_cfg)

imgs_batch = torch.randn(4, 3, 32, 32)
txt_batch  = torch.randint(0, V, (4, 8))
loss, acc  = clip(imgs_batch, txt_batch)
print(f"  CLIP loss: {loss.item():.4f}")
print(f"  Retrieval accuracy: {acc:.2%}")
print(f"  Temperature: {clip.loss_fn.temperature.item():.4f}")

# Zero-shot classification
class_tokens = [torch.randint(0, V, (1, 8)) for _ in range(3)]
probs = clip.zero_shot_classify(imgs_batch[:1], class_tokens)
print(f"  Zero-shot class probs: {[round(p.item(), 3) for p in probs]}")

# ── Perceiver Resampler ───────────────────────────────────────────────────────
print("
── Perceiver Resampler (Flamingo-style) ──")
xattn_cfg = CrossAttentionConfig(d_model=64, d_visual=16, n_heads=2,
                                   n_visual_tokens=8)
resampler = PerceiverResampler(xattn_cfg)
visual    = resampler(patch_embs)   # compress 17 patches → 8 latent tokens
print(f"  Patches in:  {tuple(patch_embs.shape)}")
print(f"  Latents out: {tuple(visual.shape)}")

# ── Visual Projector ──────────────────────────────────────────────────────────
print("
── Visual Projector (LLaVA-style) ──")
projector  = VisualProjector(d_visual=16, d_model=64, n_layers=2)
lm_tokens  = projector(patch_embs)
print(f"  Visual → LM tokens: {tuple(patch_embs.shape)} → {tuple(lm_tokens.shape)}")

# ── Cross-Modal Attention ──────────────────────────────────────────────────────
print("
── Cross-Modal Attention ──")
xattn = CrossModalAttention(xattn_cfg)
text_h = torch.randn(2, 8, 64)
fused  = xattn(text_h, patch_embs)
print(f"  Text: {tuple(text_h.shape)}, Visual: {tuple(patch_embs.shape)}")
print(f"  Fused: {tuple(fused.shape)}")

# ── VQA ───────────────────────────────────────────────────────────────────────
print("
── Visual Question Answering ──")
vqa_cfg  = VQAConfig(d_visual=16, d_model=64, vocab_size=V)
vqa_lm   = TinyLM(d_in=32, d=64)

# LLaVA-style
early_vqa = EarlyFusionVQA(encoder, vqa_lm, vqa_cfg)
logits    = early_vqa(imgs_batch[:2], torch.randint(0, V, (2, 6)))
print(f"  LLaVA VQA logits: {tuple(logits.shape)}")
answer_tokens = early_vqa.generate_answer(imgs_batch[:1], torch.randint(0, V, (1, 6)), max_len=5)
print(f"  Generated answer tokens: {answer_tokens}")

# Flamingo-style
cross_vqa = CrossAttentionVQA(encoder, vqa_lm, vqa_cfg)
logits2   = cross_vqa(imgs_batch[:2], torch.randint(0, V, (2, 6)))
print(f"  Flamingo VQA logits: {tuple(logits2.shape)}")

preds = ["cat", "2", "paris", "red", "yes"]
refs  = ["cat", "3", "paris", "blue", "yes"]
print(f"  VQA accuracy: {vqa_accuracy(preds, refs):.2%}")

# ── Image Captioning ──────────────────────────────────────────────────────────
print("
── Image Captioning ──")
cap_cfg   = CaptioningConfig(d_visual=16, d_model=64, vocab_size=V)
cap_lm    = TinyLM(d_in=32, d=64)
captioner = ImageCaptioner(encoder, cap_lm, cap_cfg)

tgt_ids  = torch.randint(1, V-1, (2, 8))
loss     = captioner.training_step(imgs_batch[:2], tgt_ids)
caption  = captioner.generate(imgs_batch[:1], max_new_tokens=8)
print(f"  Captioning loss: {loss.item():.4f}")
print(f"  Generated tokens: {caption}")

# BLEU score
candidate  = "a black cat sitting on a mat".split()
references = [
    "a cat sitting on a mat".split(),
    "black cat on mat".split(),
]
bleu = bleu_score(candidate, references, max_n=4)
print(f"  BLEU-4: {bleu:.4f}")

# ── Cross-Modal Retrieval ──────────────────────────────────────────────────────
print("
── Cross-Modal Retrieval ──")
N     = 8
imgs_idx  = torch.randn(N, 3, 32, 32)
txt_idx   = torch.randint(0, V, (N, 8))

retriever = CrossModalRetriever(clip)
retriever.index(imgs_idx, txt_idx)

i2t_results = retriever.image_to_text(imgs_idx[:1], k=3)
print(f"  I2T top-3: {[r.to_dict() for r in i2t_results]}")

t2i_results = retriever.text_to_image(txt_idx[:1], k=3)
print(f"  T2I top-3 indices: {[r.result_idx for r in t2i_results]}")

metrics = retriever.evaluate(imgs_idx[:6], txt_idx[:6])
print(f"  Retrieval metrics: {metrics.to_dict()}")

print("
VLM demo complete!")
