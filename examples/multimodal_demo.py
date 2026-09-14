"""
examples/multimodal_demo.py — NanoMind Vision-Language demo.

Demonstrates:
  1. Image patching (ViT-style)
  2. VisionEncoder: image → visual tokens
  3. MLPProjector: visual tokens → LLM space
  4. PrefixFusion: prepend visual prefix to text
  5. CrossAttentionFusion: text attends to visual tokens
  6. VisionLanguageModel: full encode_image pipeline
  7. Image augmentation

Usage:
    python examples/multimodal_demo.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.multimodal import (
    ModalityConfig, patchify, unpatchify, ImageNormalizer,
    VisionEncoder, MLPProjector, PoolingProjector,
    PrefixFusion, CrossAttentionFusion,
    VisionLanguageModel,
    ImageInput, MultimodalInput,
    ImageAugmentor,
)

# ── Tiny LM stub ──────────────────────────────────────────────────────────────
class TinyLM(nn.Module):
    def __init__(self, V=32, D=64, T=32):
        super().__init__()
        self.T = T
        self.tok_emb = nn.Embedding(V, D)
        self.lm_head = nn.Linear(D, V)
    def forward(self, x, t=None):
        h = self.tok_emb(x)
        logits = self.lm_head(h)
        return logits, None

LM_DIM = 64

print("=" * 60)
print("NanoMind Vision-Language Demo")
print("=" * 60)

# ── ModalityConfig ────────────────────────────────────────────────────────────
cfg = ModalityConfig(image_size=32, patch_size=8, vision_dim=32,
                     n_visual_tokens=8, fusion="prefix")
print(f"
ModalityConfig:")
print(f"  image={cfg.image_size}×{cfg.image_size}, patch={cfg.patch_size}×{cfg.patch_size}")
print(f"  n_patches={cfg.n_patches}, patch_dim={cfg.patch_dim}")

# ── Patchify ──────────────────────────────────────────────────────────────────
image  = torch.rand(1, 3, 32, 32)
patches = patchify(image, patch_size=8)
print(f"
Patchify: {tuple(image.shape)} → {tuple(patches.shape)}")
recon  = unpatchify(patches, patch_size=8, image_size=32)
print(f"Unpatchify: {tuple(recon.shape)} | match={torch.allclose(image, recon)}")

# ── ImageNormalizer ───────────────────────────────────────────────────────────
norm = ImageNormalizer()
normalised = norm(image)
print(f"
Normalized range: [{normalised.min():.3f}, {normalised.max():.3f}]")

# ── VisionEncoder ─────────────────────────────────────────────────────────────
encoder = VisionEncoder(cfg, n_layers=2, n_heads=4)
visual  = encoder(image)
print(f"
VisionEncoder: {tuple(image.shape)} → {tuple(visual.shape)}")
print(f"  n_params: {encoder.n_params:,}")

# ── Projectors ────────────────────────────────────────────────────────────────
mlp_proj  = MLPProjector(vision_dim=32, lm_dim=LM_DIM)
pool_proj = PoolingProjector(vision_dim=32, lm_dim=LM_DIM, target_tokens=8)
mlp_out   = mlp_proj(visual)
pool_out  = pool_proj(visual)
print(f"
MLPProjector:     {tuple(visual.shape)} → {tuple(mlp_out.shape)}")
print(f"PoolingProjector: {tuple(visual.shape)} → {tuple(pool_out.shape)}")

# ── Fusion ────────────────────────────────────────────────────────────────────
prefix_fusion = PrefixFusion(lm_embed_dim=LM_DIM)
text_emb      = torch.randn(1, 10, LM_DIM)
fused         = prefix_fusion(pool_out, text_emb)
print(f"
PrefixFusion: vis={tuple(pool_out.shape)} + txt={tuple(text_emb.shape)} → {tuple(fused.shape)}")

cross_fusion  = CrossAttentionFusion(lm_embed_dim=LM_DIM, n_heads=4)
cross_out     = cross_fusion(pool_out, text_emb)
print(f"CrossAttnFusion: → {tuple(cross_out.shape)}")

# ── VisionLanguageModel ───────────────────────────────────────────────────────
lm  = TinyLM()
vlm = VisionLanguageModel(lm, cfg, lm_dim=LM_DIM)
img = ImageInput.random(size=32)
vis_tokens = vlm.encode_image(img)
print(f"
VisionLanguageModel.encode_image: → {tuple(vis_tokens.shape)}")
print(f"  n_params: {vlm.n_params}")

# ── Augmentation ──────────────────────────────────────────────────────────────
aug   = ImageAugmentor(image_size=32, flip=True, crop=True, color=True)
pixel = torch.rand(3, 32, 32)
out   = aug(pixel)
print(f"
ImageAugmentor: {tuple(pixel.shape)} → {tuple(out.shape)}")
print("
Multimodal demo complete!")
