"""
day55_commits.py — 20 atomic commits for Day 55: Vision-Language Models (VLMs).
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== DAY 55: Vision-Language Models — 20 commits, v5.5.0 ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — vlm package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/vlm/__init__.py",
      '"""NanoMind VLM sub-package — Vision-Language Models."""\n')
commit("feat: add nanomind/vlm/ package skeleton for Vision-Language Models")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — Vision encoder (ViT-style patch embeddings)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/vlm/vision_encoder.py", '''\
"""
nanomind/vlm/vision_encoder.py — Vision Transformer (ViT) image encoder.

## Vision Transformer (Dosovitskiy et al., 2020)

Instead of convolutions, ViT treats an image as a sequence of patches:
  1. Divide 224×224 image into 16×16 patches → 196 patches
  2. Flatten each patch: 16×16×3 = 768 values
  3. Project to embedding dim D via linear layer
  4. Prepend [CLS] token (used as image representation)
  5. Add 2D positional embeddings
  6. Pass through Transformer encoder
  7. Use [CLS] output as image embedding

Patch sizes:
  ViT-B/16:  16×16 patches, 12 layers, D=768  → 196 patches
  ViT-B/32:  32×32 patches, 12 layers, D=768  → 49 patches (faster)
  ViT-L/14:  14×14 patches, 24 layers, D=1024 → 256 patches (CLIP quality)

## CLIP Vision Encoder

CLIP (Contrastive Language-Image Pre-Training) uses a ViT backbone
trained with contrastive loss to align image and text embeddings.

Input:  224×224×3 image
Output: 512-dim or 768-dim image embedding

Reference:
  Dosovitskiy et al. (2020) "An Image is Worth 16×16 Words"
  https://arxiv.org/abs/2010.11929
  Radford et al. (2021) CLIP: https://arxiv.org/abs/2103.00020
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class VisionEncoderConfig:
    """Configuration for the ViT vision encoder."""
    image_size:   int   = 224
    patch_size:   int   = 16
    in_channels:  int   = 3
    d_model:      int   = 64    # embedding dim (768 in ViT-B)
    n_layers:     int   = 4     # transformer layers (12 in ViT-B)
    n_heads:      int   = 4     # attention heads (12 in ViT-B)
    d_ff:         int   = 128   # FFN dim (3072 in ViT-B)
    dropout:      float = 0.1
    d_embed:      int   = 32    # final projection dim for CLIP

    @property
    def n_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        return self.patch_size * self.patch_size * self.in_channels

    @property
    def seq_len(self) -> int:
        return self.n_patches + 1  # +1 for CLS token


class PatchEmbedding(nn.Module):
    """
    Split image into patches and project to embedding dimension.

    Args:
        cfg: :class:`VisionEncoderConfig`.

    Example::

        embed = PatchEmbedding(cfg)
        x     = torch.randn(2, 3, 224, 224)
        out   = embed(x)   # (2, 197, d_model) — 196 patches + CLS
    """

    def __init__(self, cfg: VisionEncoderConfig) -> None:
        super().__init__()
        self.cfg        = cfg
        self.patch_proj = nn.Linear(cfg.patch_dim, cfg.d_model)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_embed  = nn.Parameter(
            torch.zeros(1, cfg.seq_len, cfg.d_model)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Split (B, C, H, W) into (B, N, patch_dim) patches."""
        B, C, H, W = x.shape
        P = self.cfg.patch_size
        # Reshape into patches
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 1, 3, 5)   # (B, H/P, W/P, C, P, P)
        x = x.reshape(B, -1, C * P * P)   # (B, N, patch_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed image patches.

        Args:
            x: Image tensor ``(B, C, H, W)``.

        Returns:
            ``(B, N+1, d_model)`` — patches + CLS token with pos embeddings.
        """
        B = x.shape[0]
        patches = self._patchify(x)                  # (B, N, patch_dim)
        tokens  = self.patch_proj(patches)            # (B, N, d_model)
        cls     = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        tokens  = torch.cat([cls, tokens], dim=1)     # (B, N+1, d_model)
        return tokens + self.pos_embed


class VisionTransformerBlock(nn.Module):
    """Single ViT transformer block (Pre-LN style)."""

    def __init__(self, cfg: VisionEncoderConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn  = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff    = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """
    ViT-style vision encoder for VLM.

    Encodes images into patch sequences and a global [CLS] embedding.

    Args:
        cfg: :class:`VisionEncoderConfig`.

    Example::

        cfg     = VisionEncoderConfig(image_size=32, patch_size=8,
                                       d_model=64, n_layers=2, d_embed=32)
        encoder = VisionEncoder(cfg)
        imgs    = torch.randn(2, 3, 32, 32)
        cls_emb, patch_embs = encoder(imgs)
        # cls_emb:    (2, 32)  — global image embedding (for CLIP)
        # patch_embs: (2, 17, 32) — per-patch embeddings (for VQA)
    """

    def __init__(self, cfg: VisionEncoderConfig) -> None:
        super().__init__()
        self.cfg       = cfg
        self.patch_emb = PatchEmbedding(cfg)
        self.layers    = nn.ModuleList([
            VisionTransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])
        self.norm      = nn.LayerNorm(cfg.d_model)
        self.proj      = nn.Linear(cfg.d_model, cfg.d_embed)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images.

        Args:
            x: ``(B, C, H, W)`` images.

        Returns:
            Tuple of:
              - cls_embedding: ``(B, d_embed)`` — global image vector
              - patch_embeddings: ``(B, N+1, d_embed)`` — all tokens projected
        """
        tokens = self.patch_emb(x)     # (B, N+1, d_model)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)     # (B, N+1, d_model)
        proj   = self.proj(tokens)     # (B, N+1, d_embed)

        cls_embedding   = proj[:, 0, :]    # (B, d_embed) — CLS token
        patch_embeddings = proj             # (B, N+1, d_embed)
        return cls_embedding, patch_embeddings
''')
commit("feat: add VisionEncoder (ViT), PatchEmbedding, VisionTransformerBlock, VisionEncoderConfig")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — CLIP contrastive loss
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/vlm/clip.py", '''\
"""
nanomind/vlm/clip.py — CLIP: Contrastive Language-Image Pre-Training.

## CLIP (Radford et al., 2021)

Train image and text encoders jointly so that:
  - Matching (image, text) pairs → similar embeddings (high cosine sim)
  - Non-matching pairs → dissimilar embeddings (low cosine sim)

## Contrastive Loss (InfoNCE / NT-Xent)

Given batch of N (image, text) pairs:
  - Compute N×N similarity matrix: S[i,j] = cos(img_i, txt_j)
  - Scale by temperature τ (learnable, initialized ~0.07)
  - Cross-entropy loss for image→text direction: predict correct text for each image
  - Cross-entropy loss for text→image direction: predict correct image for each text
  - Total loss = (img→txt + txt→img) / 2

S = img_emb @ txt_emb.T / τ   # (N, N) logits
loss = (CE(S, labels) + CE(S.T, labels)) / 2
labels = [0, 1, 2, ..., N-1]  # each sample is its own positive

## Zero-Shot Image Classification

After CLIP training:
  1. Embed all class names as text: "a photo of a cat", "a photo of a dog"
  2. Embed query image
  3. Find class with highest image-text similarity → zero-shot prediction!

CLIP (ViT-L/14) achieves 76.2% top-1 on ImageNet zero-shot!

## Applications
  - Image search (embed image, search text descriptions)
  - Image captioning (use CLIP features to condition LLM)
  - DALL-E (CLIP guides diffusion model)
  - LLaVA (CLIP vision encoder + LLaMA)

Reference:
  Radford et al. (2021) "Learning Transferable Visual Models From Natural Language Supervision"
  https://arxiv.org/abs/2103.00020
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class CLIPConfig:
    """Configuration for CLIP model."""
    d_embed:        int   = 32     # shared embedding dimension
    temperature_init: float = 0.07  # initial temperature τ
    temperature_max:  float = 100.0
    learn_temperature: bool = True


class CLIPLoss(nn.Module):
    """
    Symmetric InfoNCE contrastive loss for CLIP training.

    Args:
        cfg: :class:`CLIPConfig`.

    Example::

        loss_fn   = CLIPLoss(CLIPConfig())
        img_embs  = F.normalize(torch.randn(8, 32), dim=-1)
        txt_embs  = F.normalize(torch.randn(8, 32), dim=-1)
        loss, acc = loss_fn(img_embs, txt_embs)
        print(f"Loss: {loss.item():.3f}, Accuracy: {acc:.2%}")
    """

    def __init__(self, cfg: CLIPConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CLIPConfig()
        if self.cfg.learn_temperature:
            self.log_temp = nn.Parameter(
                torch.log(torch.tensor(self.cfg.temperature_init))
            )
        else:
            self.register_buffer(
                "log_temp",
                torch.log(torch.tensor(self.cfg.temperature_init)),
            )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(max=self.cfg.temperature_max)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings:  torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Compute symmetric CLIP contrastive loss.

        Args:
            image_embeddings: ``(B, d_embed)`` L2-normalised image vectors.
            text_embeddings:  ``(B, d_embed)`` L2-normalised text vectors.

        Returns:
            Tuple of (loss, retrieval_accuracy).
        """
        B = image_embeddings.shape[0]
        labels = torch.arange(B, device=image_embeddings.device)

        # Similarity matrix: (B, B)
        logits = (image_embeddings @ text_embeddings.T) / self.temperature

        # Symmetric cross-entropy
        loss_i2t = F.cross_entropy(logits,   labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss     = (loss_i2t + loss_t2i) / 2.0

        # Retrieval accuracy: fraction of correct top-1
        with torch.no_grad():
            acc_i2t = (logits.argmax(dim=1) == labels).float().mean()
            acc_t2i = (logits.T.argmax(dim=1) == labels).float().mean()
            acc     = ((acc_i2t + acc_t2i) / 2).item()

        return loss, acc


class CLIPModel(nn.Module):
    """
    Full CLIP model: vision encoder + text encoder + contrastive loss.

    Args:
        vision_encoder: :class:`VisionEncoder`.
        text_encoder:   LM model that returns hidden states.
        d_text:         Text encoder hidden dim.
        cfg:            :class:`CLIPConfig`.

    Example::

        clip   = CLIPModel(vision_enc, text_enc, d_text=64, cfg=CLIPConfig())
        imgs   = torch.randn(4, 3, 32, 32)
        txt    = torch.randint(0, 32, (4, 8))
        loss, acc = clip(imgs, txt)
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder:   nn.Module,
        d_text:         int,
        cfg:            CLIPConfig | None = None,
    ) -> None:
        super().__init__()
        self.vision_enc  = vision_encoder
        self.text_enc    = text_encoder
        self.cfg         = cfg or CLIPConfig()
        self.loss_fn     = CLIPLoss(self.cfg)

        # Text projection to shared embedding space
        self.text_proj = nn.Linear(d_text, self.cfg.d_embed)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to normalised d_embed vectors."""
        cls_emb, _ = self.vision_enc(images)
        return F.normalize(cls_emb, dim=-1)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode token IDs to normalised d_embed vectors."""
        out    = self.text_enc(input_ids)
        hidden = out[0] if isinstance(out, tuple) else out
        # Use last token (or mean pool)
        pooled = hidden.mean(dim=1)          # (B, d_text)
        proj   = self.text_proj(pooled)      # (B, d_embed)
        return F.normalize(proj, dim=-1)

    def forward(
        self,
        images:    torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Forward pass: compute embeddings and CLIP loss.

        Returns:
            (loss, retrieval_accuracy).
        """
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(input_ids)
        return self.loss_fn(img_emb, txt_emb)

    @torch.no_grad()
    def zero_shot_classify(
        self,
        image:       torch.Tensor,
        class_prompts: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Zero-shot image classification.

        Args:
            image:         ``(1, C, H, W)`` image.
            class_prompts: List of token ID tensors (one per class).

        Returns:
            ``(n_classes,)`` softmax probabilities.
        """
        img_emb  = self.encode_image(image)    # (1, d_embed)
        txt_embs = torch.stack([
            self.encode_text(p) for p in class_prompts
        ]).squeeze(1)                          # (n_classes, d_embed)
        logits   = (img_emb @ txt_embs.T) / self.loss_fn.temperature
        return F.softmax(logits[0], dim=-1)

    def image_text_similarity(
        self,
        image:    torch.Tensor,
        text_ids: torch.Tensor,
    ) -> float:
        """Cosine similarity between a single image and text."""
        img = self.encode_image(image.unsqueeze(0))
        txt = self.encode_text(text_ids.unsqueeze(0))
        return (img * txt).sum().item()
''')
commit("feat: add CLIPLoss (InfoNCE, symmetric), CLIPModel (encode_image/text, zero_shot_classify)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — Cross-modal attention (for VQA / captioning)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/vlm/cross_attention.py", '''\
"""
nanomind/vlm/cross_attention.py — Cross-modal attention for VLMs.

## Cross-Attention in VLMs

After encoding images with ViT, we need to fuse visual and language features.

Two main architectures:

1. Early Fusion (LLaVA style):
   - Concatenate image patch tokens + text tokens
   - Feed concatenated sequence to LLM
   - Image tokens act like extra "visual tokens" in the context
   - Simple but requires fine-tuning the full LLM

2. Cross-Attention Fusion (Flamingo style):
   - Keep image and text in separate streams
   - Add cross-attention layers to LLM: text attends to image features
   - More efficient: can freeze the LLM
   - Better for long image sequences

## LLaVA Architecture (Visual Instruction Tuning)

  image → ViT encoder → MLP projector → visual tokens
  question → tokenizer → text tokens
  [visual tokens] + [text tokens] → LLaMA → answer

Very simple, very effective. LLaVA-1.5 beats GPT-4V on many benchmarks!

## Flamingo Architecture (DeepMind, 2022)

  image → NFNet vision encoder → Perceiver Resampler → 64 visual tokens
  [cross-attention layers inserted into frozen LM every K layers]
  → few-shot multimodal QA

References:
  Liu et al. (2023) LLaVA: https://arxiv.org/abs/2304.08485
  Alayrac et al. (2022) Flamingo: https://arxiv.org/abs/2204.14198
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class CrossAttentionConfig:
    """Configuration for cross-modal attention."""
    d_model:   int   = 64    # LM hidden dim
    d_visual:  int   = 32    # vision encoder output dim
    n_heads:   int   = 4
    dropout:   float = 0.1
    n_visual_tokens: int = 64  # Perceiver: compress to this many tokens


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler: compress variable-length image patches to
    fixed-size visual token sequence (Flamingo-style).

    Args:
        cfg: :class:`CrossAttentionConfig`.

    Example::

        resampler = PerceiverResampler(cfg)
        patches   = torch.randn(2, 197, 32)   # ViT output
        visual    = resampler(patches)          # (2, 64, 32)
    """

    def __init__(self, cfg: CrossAttentionConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        # Learnable latent visual tokens (the "queries" in Perceiver)
        self.latents = nn.Parameter(
            torch.randn(1, cfg.n_visual_tokens, cfg.d_visual)
        )
        self.attn = nn.MultiheadAttention(
            cfg.d_visual, cfg.n_heads,
            dropout=cfg.dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(cfg.d_visual)
        self.norm_kv = nn.LayerNorm(cfg.d_visual)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_visual, cfg.d_visual * 4),
            nn.GELU(),
            nn.Linear(cfg.d_visual * 4, cfg.d_visual),
        )
        self.norm_ff = nn.LayerNorm(cfg.d_visual)

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compress image tokens to fixed-size visual tokens.

        Args:
            image_tokens: ``(B, N, d_visual)`` from vision encoder.

        Returns:
            ``(B, n_visual_tokens, d_visual)`` compressed visual tokens.
        """
        B    = image_tokens.shape[0]
        q    = self.latents.expand(B, -1, -1)   # (B, n_vis, d_visual)
        kv   = image_tokens

        q_n  = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        out, _ = self.attn(q_n, kv_n, kv_n)
        q    = q + out
        q    = q + self.ff(self.norm_ff(q))
        return q


class VisualProjector(nn.Module):
    """
    Linear MLP projector from vision dim → LM dim (LLaVA-style).

    Projects ViT output to LM's token embedding space so visual
    tokens can be concatenated with text tokens.

    Args:
        d_visual: Vision encoder output dim.
        d_model:  LM embedding dim.
        n_layers: Number of MLP layers (LLaVA uses 2).

    Example::

        proj   = VisualProjector(d_visual=32, d_model=64, n_layers=2)
        visual = torch.randn(2, 17, 32)   # patch embeddings
        lm_vis = proj(visual)              # (2, 17, 64) — same dim as LM
    """

    def __init__(self, d_visual: int, d_model: int, n_layers: int = 2) -> None:
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_dim  = d_visual if i == 0 else d_model
            layers.append(nn.Linear(in_dim, d_model))
            if i < n_layers - 1:
                layers.append(nn.GELU())
        self.proj = nn.Sequential(*layers)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """Project visual tokens to LM embedding space."""
        return self.proj(visual_tokens)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: text tokens attend to visual tokens.

    Used in Flamingo-style architectures where cross-attention layers
    are inserted into the LM at regular intervals.

    Args:
        cfg: :class:`CrossAttentionConfig`.

    Example::

        xattn   = CrossModalAttention(cfg)
        text_h  = torch.randn(2, 12, 64)   # LM hidden states
        visual  = torch.randn(2, 64, 32)   # visual tokens
        out     = xattn(text_h, visual)     # (2, 12, 64)
    """

    def __init__(self, cfg: CrossAttentionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Project visual to d_model for cross-attention keys/values
        self.visual_proj = nn.Linear(cfg.d_visual, cfg.d_model)
        self.cross_attn  = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads,
            dropout=cfg.dropout, batch_first=True
        )
        self.norm_q  = nn.LayerNorm(cfg.d_model)
        self.norm_kv = nn.LayerNorm(cfg.d_model)
        self.gate    = nn.Parameter(torch.zeros(1))   # gating (tanh)

    def forward(
        self,
        text_hidden:   torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attend text hidden states to visual tokens.

        Args:
            text_hidden:   ``(B, T, d_model)`` LM hidden states.
            visual_tokens: ``(B, N, d_visual)`` visual token sequence.

        Returns:
            ``(B, T, d_model)`` updated text hidden states.
        """
        # Project visual to d_model
        v    = self.visual_proj(visual_tokens)    # (B, N, d_model)
        q    = self.norm_q(text_hidden)
        kv   = self.norm_kv(v)
        out, _ = self.cross_attn(q, kv, kv)
        # Gated residual connection (Flamingo style)
        return text_hidden + self.gate.tanh() * out
''')
commit("feat: add PerceiverResampler, VisualProjector, CrossModalAttention — Flamingo/LLaVA fusion")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — VQA model
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/vlm/vqa.py", '''\
"""
nanomind/vlm/vqa.py — Visual Question Answering (VQA).

## VQA Task

Given: image + natural language question
Produce: natural language answer

VQA datasets:
  - VQA v2: 1.1M questions about COCO images
  - GQA: compositional questions requiring spatial/logical reasoning
  - TextVQA: questions requiring reading text in images
  - ScienceQA: multimodal science questions with explanations

## VQA Model Architectures

1. Early Fusion (LLaVA style):
   visual_tokens = ViT(image) → projector
   input = [visual_tokens] + tokenize(question)
   answer = LLM(input)

2. Encoder-Decoder:
   visual_features = ViT(image)
   encoder_output  = cross_attn(text_features, visual_features)
   answer          = decoder(encoder_output)

3. BLIP-2 (Li et al., 2023):
   image → ViT → Q-Former (32 learnable queries attend to patches) → LLM
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from nanomind.vlm.cross_attention import VisualProjector, CrossModalAttention, CrossAttentionConfig


@dataclass
class VQAConfig:
    """Configuration for the VQA model."""
    d_visual:    int   = 32    # vision encoder output dim
    d_model:     int   = 64    # LM hidden dim
    vocab_size:  int   = 256
    max_answer_len: int = 20
    fusion:      str   = "early"   # "early" | "cross_attention"


@dataclass
class VQASample:
    """A single VQA example."""
    image_id:  str
    question:  str
    answer:    str
    q_type:    str = "what"   # what/where/who/how many/yes-no

    def to_dict(self) -> dict:
        return {"image_id": self.image_id, "question": self.question,
                "answer":   self.answer, "q_type": self.q_type}


@dataclass
class VQAResult:
    """Model prediction for a VQA sample."""
    question:       str
    predicted_answer: str
    confidence:     float
    correct:        bool = False

    def to_dict(self) -> dict:
        return {"question": self.question, "predicted": self.predicted_answer,
                "confidence": round(self.confidence, 4), "correct": self.correct}


class EarlyFusionVQA(nn.Module):
    """
    LLaVA-style VQA: concatenate visual tokens + text tokens → LM.

    Args:
        vision_encoder: ViT encoder (returns cls_emb, patch_embs).
        lm:             Language model.
        cfg:            :class:`VQAConfig`.

    Example::

        model  = EarlyFusionVQA(vision_enc, lm, VQAConfig())
        imgs   = torch.randn(2, 3, 32, 32)
        q_ids  = torch.randint(0, 256, (2, 8))
        logits = model(imgs, q_ids)   # (2, n_visual+8, vocab_size)
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        lm:             nn.Module,
        cfg:            VQAConfig | None = None,
    ) -> None:
        super().__init__()
        self.vision_enc = vision_encoder
        self.lm         = lm
        self.cfg        = cfg or VQAConfig()
        self.projector  = VisualProjector(
            cfg.d_visual, cfg.d_model, n_layers=2
        )

    def forward(
        self,
        images:    torch.Tensor,
        q_ids:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: ``(B, C, H, W)`` images.
            q_ids:  ``(B, T)`` question token IDs.

        Returns:
            ``(B, N_vis + T, vocab_size)`` logits.
        """
        # 1. Encode image patches
        _, patch_embs = self.vision_enc(images)    # (B, N, d_visual)
        visual_tokens = self.projector(patch_embs) # (B, N, d_model)

        # 2. Encode text tokens
        # Get LM embeddings (first layer)
        if hasattr(self.lm, 'emb'):
            txt_tokens = self.lm.emb(q_ids % self.cfg.vocab_size)
        else:
            txt_tokens = q_ids.float().unsqueeze(-1).expand(
                *q_ids.shape, self.cfg.d_model
            )

        # 3. Concatenate: [visual | text]
        combined = torch.cat([visual_tokens, txt_tokens], dim=1)  # (B, N+T, d_model)

        # 4. Pass through LM (using GRU as LM backbone)
        if hasattr(self.lm, 'rnn'):
            h, _ = self.lm.rnn(combined)
            logits = self.lm.head(h)       # (B, N+T, vocab_size)
        else:
            out    = self.lm(combined.long().mean(dim=-1, keepdim=True).squeeze())
            logits = out[0] if isinstance(out, tuple) else out
        return logits

    @torch.no_grad()
    def generate_answer(
        self,
        images: torch.Tensor,
        q_ids:  torch.Tensor,
        max_len: int = 10,
    ) -> list[int]:
        """Greedy decoding to generate answer tokens."""
        logits = self.forward(images, q_ids)   # (B, N+T, V)
        # Take last position tokens as answer start
        generated = logits[:, -max_len:, :].argmax(dim=-1)
        return generated[0].tolist()


class CrossAttentionVQA(nn.Module):
    """
    Flamingo-style VQA: cross-attention between text and visual tokens.

    Args:
        vision_encoder: ViT encoder.
        lm:             Language model.
        cfg:            :class:`VQAConfig`.

    Example::

        model  = CrossAttentionVQA(vision_enc, lm, VQAConfig())
        imgs   = torch.randn(2, 3, 32, 32)
        q_ids  = torch.randint(0, 256, (2, 8))
        logits = model(imgs, q_ids)
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        lm:             nn.Module,
        cfg:            VQAConfig | None = None,
    ) -> None:
        super().__init__()
        self.vision_enc = vision_encoder
        self.lm         = lm
        self.cfg        = cfg or VQAConfig()
        xattn_cfg       = CrossAttentionConfig(
            d_model=cfg.d_model, d_visual=cfg.d_visual
        )
        self.cross_attn = CrossModalAttention(xattn_cfg)
        self.head       = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(
        self,
        images: torch.Tensor,
        q_ids:  torch.Tensor,
    ) -> torch.Tensor:
        _, patch_embs = self.vision_enc(images)  # (B, N, d_visual)

        # Encode text with LM
        if hasattr(self.lm, 'emb'):
            txt_emb = self.lm.emb(q_ids % self.cfg.vocab_size)
        else:
            txt_emb = torch.randn(*q_ids.shape, self.cfg.d_model)

        if hasattr(self.lm, 'rnn'):
            text_h, _ = self.lm.rnn(txt_emb)    # (B, T, d_model)
        else:
            text_h = txt_emb

        # Cross-attend text to visual
        fused = self.cross_attn(text_h, patch_embs)   # (B, T, d_model)
        return self.head(fused)                         # (B, T, vocab_size)


def vqa_accuracy(predictions: list[str], references: list[str]) -> float:
    """VQA accuracy: fraction of predictions that match any reference answer."""
    n_correct = sum(
        1 for pred, ref in zip(predictions, references)
        if pred.strip().lower() == ref.strip().lower()
    )
    return n_correct / max(len(predictions), 1)
''')
commit("feat: add EarlyFusionVQA (LLaVA-style), CrossAttentionVQA (Flamingo-style), VQASample, vqa_accuracy")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Image captioning
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/vlm/captioning.py", '''\
"""
nanomind/vlm/captioning.py — Image captioning model.

## Image Captioning

Given an image, generate a natural language description.

Architecture:
  ViT encoder → visual features → LM decoder (autoregressive) → caption

Training: teacher forcing
  Input:  [BOS] + caption tokens
  Target: caption tokens + [EOS]
  Loss:   cross-entropy on target tokens only

## BLEU Score

Standard metric for captioning quality (also used for MT):
  BLEU-n = geometric mean of n-gram precisions × brevity penalty

  Modified precision:
    p_n = # n-gram matches / # n-grams in candidate (clipped to reference count)

  Geometric mean:
    BLEU = BP × exp(Σ wₙ × log pₙ)  where wₙ = 1/4, BP = brevity penalty

BLEU-4 is the standard metric for captioning benchmarks.

## CIDEr (Consensus-Based Image Description Evaluation)

Better metric for captioning than BLEU:
  - Weights n-grams by TF-IDF (rewards distinctive words)
  - Correlates better with human judgment

## Modern Captioning Models
  - BLIP-2: ViT + Q-Former + Flan-T5
  - LLaVA-1.5: ViT-L/336 + MLP + Vicuna-13B
  - Gemini: Proprietary multimodal
"""

from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.vlm.cross_attention import VisualProjector


@dataclass
class CaptioningConfig:
    """Configuration for image captioning model."""
    d_visual:   int   = 32
    d_model:    int   = 64
    vocab_size: int   = 256
    bos_token:  int   = 1
    eos_token:  int   = 2
    pad_token:  int   = 0
    max_len:    int   = 32


class ImageCaptioner(nn.Module):
    """
    Image captioning model: ViT encoder + autoregressive LM decoder.

    Args:
        vision_encoder: ViT image encoder.
        lm:             Language model (used as decoder).
        cfg:            :class:`CaptioningConfig`.

    Example::

        captioner = ImageCaptioner(vision_enc, lm, CaptioningConfig())
        imgs      = torch.randn(2, 3, 32, 32)
        tgt_ids   = torch.randint(0, 256, (2, 10))
        loss      = captioner.training_step(imgs, tgt_ids)
        captions  = captioner.generate(imgs[:1], max_new_tokens=10)
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        lm:             nn.Module,
        cfg:            CaptioningConfig | None = None,
    ) -> None:
        super().__init__()
        self.vision_enc = vision_encoder
        self.lm         = lm
        self.cfg        = cfg or CaptioningConfig()
        self.projector  = VisualProjector(cfg.d_visual, cfg.d_model)

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to visual token sequence."""
        _, patch_embs = self.vision_enc(images)
        return self.projector(patch_embs)   # (B, N, d_model)

    def training_step(
        self,
        images:  torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute teacher-forced captioning loss.

        Args:
            images:  ``(B, C, H, W)`` images.
            tgt_ids: ``(B, T)`` target caption token IDs.

        Returns:
            Scalar cross-entropy loss.
        """
        visual_tokens = self._encode_image(images)   # (B, N, d_model)

        # Get text embeddings
        if hasattr(self.lm, 'emb'):
            txt_emb = self.lm.emb(tgt_ids[:, :-1] % self.cfg.vocab_size)
        else:
            txt_emb = torch.zeros(*tgt_ids[:, :-1].shape, self.cfg.d_model)

        # Concat visual prefix + text input
        decoder_input = torch.cat([visual_tokens, txt_emb], dim=1)

        # Run through LM decoder
        if hasattr(self.lm, 'rnn'):
            h, _ = self.lm.rnn(decoder_input)
        else:
            h = decoder_input

        # Predict next tokens (only on text portion)
        N       = visual_tokens.shape[1]
        text_h  = h[:, N:, :]            # (B, T-1, d_model)

        if hasattr(self.lm, 'head'):
            logits = self.lm.head(text_h)  # (B, T-1, V)
        else:
            logits = text_h

        target = tgt_ids[:, 1:].clamp(0, self.cfg.vocab_size - 1)
        loss   = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            target.reshape(-1),
        )
        return loss

    @torch.no_grad()
    def generate(
        self,
        images:         torch.Tensor,
        max_new_tokens: int = 20,
    ) -> list[int]:
        """Greedy decoding to generate a caption."""
        visual_tokens = self._encode_image(images)   # (1, N, d_model)
        generated     = [self.cfg.bos_token]

        for _ in range(max_new_tokens):
            tgt = torch.tensor([generated]).long()
            if hasattr(self.lm, 'emb'):
                txt_emb = self.lm.emb(tgt % self.cfg.vocab_size)
            else:
                txt_emb = torch.zeros(1, len(generated), self.cfg.d_model)

            dec_in = torch.cat([visual_tokens, txt_emb], dim=1)
            if hasattr(self.lm, 'rnn'):
                h, _ = self.lm.rnn(dec_in)
            else:
                h = dec_in

            last_h = h[:, -1, :]  # (1, d_model)
            if hasattr(self.lm, 'head'):
                logits = self.lm.head(last_h)   # (1, V)
            else:
                logits = torch.randn(1, self.cfg.vocab_size)

            next_token = logits.argmax(dim=-1).item()
            generated.append(next_token)
            if next_token == self.cfg.eos_token:
                break

        return generated


def bleu_score(
    candidate: list[str],
    references: list[list[str]],
    max_n:     int = 4,
) -> float:
    """
    Compute BLEU-n score.

    Args:
        candidate:  Tokenised candidate (list of words).
        references: List of tokenised references.
        max_n:      Maximum n-gram order.

    Returns:
        BLEU score in [0, 1].
    """
    if not candidate:
        return 0.0

    # Brevity penalty
    ref_len  = min(len(r) for r in references)
    cand_len = len(candidate)
    bp       = 1.0 if cand_len >= ref_len else math.exp(1 - ref_len / cand_len)

    # Modified n-gram precision
    log_sum = 0.0
    for n in range(1, max_n + 1):
        cand_ngrams = Counter(tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1))
        if not cand_ngrams:
            return 0.0
        clipped = 0
        for ngram, count in cand_ngrams.items():
            max_ref = max(
                Counter(tuple(r[i:i+n]) for i in range(len(r)-n+1)).get(ngram, 0)
                for r in references
            )
            clipped += min(count, max_ref)
        total = sum(cand_ngrams.values())
        if total == 0 or clipped == 0:
            return 0.0
        log_sum += math.log(clipped / total) / max_n

    return bp * math.exp(log_sum)
''')
commit("feat: add ImageCaptioner (teacher-forcing, greedy generate), bleu_score (BLEU-4), CaptioningConfig")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Cross-modal retrieval
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/vlm/retrieval.py", '''\
"""
nanomind/vlm/retrieval.py — Cross-modal retrieval (image←→text search).

## Cross-Modal Retrieval Tasks

Image-to-Text Retrieval (I2T):
  Given an image, find the matching caption from a pool.
  "Retrieve the 5 most relevant captions for this image."

Text-to-Image Retrieval (T2I):
  Given a caption, find the matching image from a pool.
  "Find images showing a dog playing on a beach."

## Evaluation Metrics

Recall@K (R@K):
  "Is the correct match in the top K results?"
  R@1 = fraction of queries where correct is #1
  R@5 = fraction of queries where correct is in top 5
  R@10 = fraction of queries where correct is in top 10

Median Rank:
  Median rank of the correct item across all queries.
  Lower = better.

## CLIP for Retrieval

CLIP produces aligned image-text embeddings:
  1. Pre-encode all images/texts into embedding vectors
  2. At query time: embed query → cosine similarity search
  3. Return top-K matches

This enables O(1) per-query retrieval (after O(N) indexing).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    query_idx:  int
    result_idx: int
    score:      float
    rank:       int

    def to_dict(self) -> dict:
        return {"query": self.query_idx, "result": self.result_idx,
                "score": round(self.score, 4), "rank": self.rank}


@dataclass
class RetrievalMetrics:
    """Recall@K metrics for retrieval evaluation."""
    r_at_1:    float
    r_at_5:    float
    r_at_10:   float
    median_rank: float
    n_queries:  int

    def to_dict(self) -> dict:
        return {
            "R@1":         round(self.r_at_1, 4),
            "R@5":         round(self.r_at_5, 4),
            "R@10":        round(self.r_at_10, 4),
            "median_rank": self.median_rank,
            "n_queries":   self.n_queries,
        }


class CrossModalRetriever:
    """
    Cross-modal retrieval using CLIP-style embeddings.

    Supports:
      - Image-to-Text (I2T): query image → find matching text
      - Text-to-Image (T2I): query text → find matching image

    Args:
        clip_model: Trained :class:`CLIPModel`.

    Example::

        retriever = CrossModalRetriever(clip_model)
        retriever.index(images, text_ids)

        # Find top-5 captions for a query image
        results = retriever.image_to_text(query_image, k=5)
        # Find top-5 images for a query text
        results = retriever.text_to_image(query_text_ids, k=5)
    """

    def __init__(self, clip_model) -> None:
        self.clip = clip_model
        self._image_embs: torch.Tensor | None = None
        self._text_embs:  torch.Tensor | None = None
        self._n_items: int = 0

    @torch.no_grad()
    def index(
        self,
        images:    torch.Tensor,
        text_ids:  torch.Tensor,
        batch_size: int = 32,
    ) -> None:
        """
        Pre-encode and index all image-text pairs.

        Args:
            images:    ``(N, C, H, W)`` images.
            text_ids:  ``(N, T)`` text token IDs.
            batch_size: Batch size for encoding.
        """
        img_embs = []
        txt_embs = []
        N = images.shape[0]
        for i in range(0, N, batch_size):
            ib = images[i:i+batch_size]
            tb = text_ids[i:i+batch_size]
            img_embs.append(self.clip.encode_image(ib))
            txt_embs.append(self.clip.encode_text(tb))
        self._image_embs = torch.cat(img_embs, dim=0)
        self._text_embs  = torch.cat(txt_embs, dim=0)
        self._n_items    = N

    @torch.no_grad()
    def image_to_text(
        self,
        query_image: torch.Tensor,
        k:           int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve top-k texts for a query image."""
        if self._text_embs is None:
            raise RuntimeError("Call index() first")
        q    = self.clip.encode_image(query_image)          # (1, d)
        sims = (q @ self._text_embs.T).squeeze(0)          # (N,)
        return self._top_k(sims, k, query_idx=0)

    @torch.no_grad()
    def text_to_image(
        self,
        query_text: torch.Tensor,
        k:          int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve top-k images for a query text."""
        if self._image_embs is None:
            raise RuntimeError("Call index() first")
        q    = self.clip.encode_text(query_text)            # (1, d)
        sims = (q @ self._image_embs.T).squeeze(0)         # (N,)
        return self._top_k(sims, k, query_idx=0)

    def _top_k(
        self,
        scores:    torch.Tensor,
        k:         int,
        query_idx: int,
    ) -> list[RetrievalResult]:
        k = min(k, len(scores))
        top_scores, top_idx = scores.topk(k)
        return [
            RetrievalResult(
                query_idx  = query_idx,
                result_idx = idx.item(),
                score      = score.item(),
                rank       = rank,
            )
            for rank, (idx, score) in enumerate(
                zip(top_idx.tolist(), top_scores.tolist())
            )
        ]

    def evaluate(
        self,
        images:   torch.Tensor,
        text_ids: torch.Tensor,
        ks:       list[int] | None = None,
    ) -> RetrievalMetrics:
        """
        Evaluate I2T and T2I recall@K.

        Args:
            images:   ``(N, C, H, W)`` images.
            text_ids: ``(N, T)`` matching text IDs.
            ks:       List of K values for Recall@K.

        Returns:
            :class:`RetrievalMetrics`.
        """
        ks = ks or [1, 5, 10]
        self.index(images, text_ids)

        N    = self._n_items
        ranks_i2t = []
        ranks_t2i = []

        for i in range(N):
            # I2T: rank of correct text for image i
            q    = self._image_embs[i:i+1]
            sims = (q @ self._text_embs.T).squeeze(0)
            rank = (sims > sims[i]).sum().item()
            ranks_i2t.append(rank)

            # T2I: rank of correct image for text i
            q    = self._text_embs[i:i+1]
            sims = (q @ self._image_embs.T).squeeze(0)
            rank = (sims > sims[i]).sum().item()
            ranks_t2i.append(rank)

        all_ranks = ranks_i2t + ranks_t2i

        def recall_at(ranks, k):
            return sum(1 for r in ranks if r < k) / len(ranks)

        return RetrievalMetrics(
            r_at_1      = recall_at(all_ranks, 1),
            r_at_5      = recall_at(all_ranks, 5),
            r_at_10     = recall_at(all_ranks, 10),
            median_rank = float(sorted(all_ranks)[len(all_ranks) // 2] + 1),
            n_queries   = N,
        )
''')
commit("feat: add CrossModalRetriever — index, image_to_text, text_to_image, evaluate (Recall@K metrics)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — vlm __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/vlm/__init__.py", '''\
"""NanoMind VLM sub-package — Vision-Language Models.

Implements the full VLM stack:
  1. VisionEncoderConfig       — image size, patch size, d_model, n_layers
  2. PatchEmbedding            — patchify + project + CLS token + pos embed
  3. VisionTransformerBlock    — Pre-LN self-attention + FFN
  4. VisionEncoder             — full ViT: embed → transformer → project
  5. CLIPConfig                — temperature, d_embed
  6. CLIPLoss                  — symmetric InfoNCE contrastive loss
  7. CLIPModel                 — encode_image/text, zero_shot_classify
  8. CrossAttentionConfig      — d_model, d_visual, n_heads
  9. PerceiverResampler        — compress N patches to fixed visual tokens
  10. VisualProjector          — MLP projector (ViT dim → LM dim)
  11. CrossModalAttention      — text attends to visual tokens (Flamingo)
  12. VQAConfig / VQASample    — VQA data structures
  13. EarlyFusionVQA           — LLaVA-style (visual + text tokens concatenated)
  14. CrossAttentionVQA        — Flamingo-style (cross-attention layers)
  15. vqa_accuracy             — VQA evaluation metric
  16. CaptioningConfig         — captioning model config
  17. ImageCaptioner           — ViT + LM decoder, training_step, generate
  18. bleu_score               — BLEU-4 captioning metric
  19. CrossModalRetriever      — image/text search with CLIP embeddings
  20. RetrievalMetrics         — Recall@1/5/10, median rank

Primary exports:
    - :class:`VisionEncoderConfig`   — ViT configuration
    - :class:`PatchEmbedding`        — image → patch tokens
    - :class:`VisionEncoder`         — full ViT encoder
    - :class:`CLIPLoss`              — symmetric contrastive loss
    - :class:`CLIPModel`             — full CLIP (encode + loss)
    - :class:`PerceiverResampler`    — Flamingo-style token compression
    - :class:`VisualProjector`       — LLaVA-style MLP projector
    - :class:`CrossModalAttention`   — text-visual cross-attention
    - :class:`EarlyFusionVQA`        — LLaVA-style VQA model
    - :class:`CrossAttentionVQA`     — Flamingo-style VQA model
    - :func:`vqa_accuracy`           — VQA eval metric
    - :class:`ImageCaptioner`        — image → caption model
    - :func:`bleu_score`             — BLEU-4 metric
    - :class:`CrossModalRetriever`   — image/text retrieval + Recall@K
    - :class:`RetrievalMetrics`      — R@1, R@5, R@10, median rank
"""

from nanomind.vlm.vision_encoder import (
    VisionEncoderConfig, PatchEmbedding, VisionTransformerBlock, VisionEncoder,
)
from nanomind.vlm.clip import CLIPConfig, CLIPLoss, CLIPModel
from nanomind.vlm.cross_attention import (
    CrossAttentionConfig, PerceiverResampler, VisualProjector, CrossModalAttention,
)
from nanomind.vlm.vqa import (
    VQAConfig, VQASample, VQAResult, EarlyFusionVQA, CrossAttentionVQA, vqa_accuracy,
)
from nanomind.vlm.captioning import CaptioningConfig, ImageCaptioner, bleu_score
from nanomind.vlm.retrieval import (
    CrossModalRetriever, RetrievalMetrics, RetrievalResult,
)

__all__ = [
    "VisionEncoderConfig", "PatchEmbedding", "VisionTransformerBlock", "VisionEncoder",
    "CLIPConfig", "CLIPLoss", "CLIPModel",
    "CrossAttentionConfig", "PerceiverResampler", "VisualProjector", "CrossModalAttention",
    "VQAConfig", "VQASample", "VQAResult", "EarlyFusionVQA", "CrossAttentionVQA",
    "vqa_accuracy",
    "CaptioningConfig", "ImageCaptioner", "bleu_score",
    "CrossModalRetriever", "RetrievalMetrics", "RetrievalResult",
]
''')
commit("refactor: export all VLM components from nanomind/vlm/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — example
# ══════════════════════════════════════════════════════════════════════════════
write("examples/vlm_demo.py", '''\
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
print("\n── Vision Encoder (ViT) ──")
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
print("\n── Patch Embedding ──")
pe  = PatchEmbedding(cfg)
out = pe(imgs)
print(f"  PatchEmbedding output: {tuple(out.shape)}  (= 1 CLS + 16 patches)")

# ── CLIP ──────────────────────────────────────────────────────────────────────
print("\n── CLIP Contrastive Loss ──")
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
print("\n── Perceiver Resampler (Flamingo-style) ──")
xattn_cfg = CrossAttentionConfig(d_model=64, d_visual=16, n_heads=2,
                                   n_visual_tokens=8)
resampler = PerceiverResampler(xattn_cfg)
visual    = resampler(patch_embs)   # compress 17 patches → 8 latent tokens
print(f"  Patches in:  {tuple(patch_embs.shape)}")
print(f"  Latents out: {tuple(visual.shape)}")

# ── Visual Projector ──────────────────────────────────────────────────────────
print("\n── Visual Projector (LLaVA-style) ──")
projector  = VisualProjector(d_visual=16, d_model=64, n_layers=2)
lm_tokens  = projector(patch_embs)
print(f"  Visual → LM tokens: {tuple(patch_embs.shape)} → {tuple(lm_tokens.shape)}")

# ── Cross-Modal Attention ──────────────────────────────────────────────────────
print("\n── Cross-Modal Attention ──")
xattn = CrossModalAttention(xattn_cfg)
text_h = torch.randn(2, 8, 64)
fused  = xattn(text_h, patch_embs)
print(f"  Text: {tuple(text_h.shape)}, Visual: {tuple(patch_embs.shape)}")
print(f"  Fused: {tuple(fused.shape)}")

# ── VQA ───────────────────────────────────────────────────────────────────────
print("\n── Visual Question Answering ──")
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
print("\n── Image Captioning ──")
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
print("\n── Cross-Modal Retrieval ──")
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

print("\nVLM demo complete!")
''')
commit("feat: add examples/vlm_demo.py — ViT, CLIP, Perceiver, VQA, captioning, retrieval end-to-end")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_vlm.py", '''\
"""tests/test_vlm.py — Tests for NanoMind VLM package."""
import pytest
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

V = 32

def make_cfg():
    return VisionEncoderConfig(
        image_size=16, patch_size=4, d_model=16,
        n_layers=1, n_heads=2, d_ff=32, d_embed=8
    )

class TinyLM(nn.Module):
    def __init__(self, d_in=16, d=16):
        super().__init__()
        self.emb  = nn.Embedding(V, d_in)
        self.rnn  = nn.GRU(d_in, d, batch_first=True)
        self.head = nn.Linear(d, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x % V))
        return self.head(h), None


# ── Vision Encoder ────────────────────────────────────────────────────────────

class TestVisionEncoder:
    def test_n_patches(self):
        cfg = make_cfg()
        assert cfg.n_patches == (16 // 4) ** 2  # 16

    def test_patch_dim(self):
        cfg = make_cfg()
        assert cfg.patch_dim == 4 * 4 * 3  # 48

    def test_encoder_cls_shape(self):
        cfg     = make_cfg()
        encoder = VisionEncoder(cfg)
        imgs    = torch.randn(2, 3, 16, 16)
        cls, patches = encoder(imgs)
        assert cls.shape == (2, 8)         # d_embed=8

    def test_encoder_patch_shape(self):
        cfg     = make_cfg()
        encoder = VisionEncoder(cfg)
        imgs    = torch.randn(2, 3, 16, 16)
        _, patches = encoder(imgs)
        assert patches.shape == (2, cfg.n_patches + 1, 8)   # +1 CLS

    def test_patch_embedding_output(self):
        cfg = make_cfg()
        pe  = PatchEmbedding(cfg)
        x   = torch.randn(2, 3, 16, 16)
        out = pe(x)
        assert out.shape == (2, cfg.seq_len, cfg.d_model)


# ── CLIP ──────────────────────────────────────────────────────────────────────

class TestCLIP:
    def _clip(self):
        cfg     = make_cfg()
        enc     = VisionEncoder(cfg)
        lm      = TinyLM()
        clip_cfg = CLIPConfig(d_embed=8)
        return CLIPModel(enc, lm, d_text=16, cfg=clip_cfg)

    def test_loss_scalar(self):
        clip = self._clip()
        imgs = torch.randn(4, 3, 16, 16)
        txt  = torch.randint(0, V, (4, 6))
        loss, acc = clip(imgs, txt)
        assert loss.item() > 0
        assert 0.0 <= acc <= 1.0

    def test_encode_image_shape(self):
        clip = self._clip()
        imgs = torch.randn(3, 3, 16, 16)
        emb  = clip.encode_image(imgs)
        assert emb.shape == (3, 8)

    def test_encode_image_normalised(self):
        clip = self._clip()
        imgs = torch.randn(2, 3, 16, 16)
        emb  = clip.encode_image(imgs)
        norms = emb.norm(dim=-1)
        assert all(abs(n.item() - 1.0) < 0.01 for n in norms)

    def test_zero_shot_classify(self):
        clip  = self._clip()
        img   = torch.randn(1, 3, 16, 16)
        probs = clip.zero_shot_classify(img,
                    [torch.randint(0, V, (1, 4)) for _ in range(3)])
        assert probs.shape == (3,)
        assert abs(probs.sum().item() - 1.0) < 0.01

    def test_clip_loss_symmetric(self):
        loss_fn  = CLIPLoss()
        img_embs = F.normalize(torch.randn(4, 8), dim=-1)
        txt_embs = F.normalize(torch.randn(4, 8), dim=-1)
        loss, _  = loss_fn(img_embs, txt_embs)
        assert loss.item() > 0


# ── Cross-Attention ───────────────────────────────────────────────────────────

class TestCrossAttention:
    def _cfg(self):
        return CrossAttentionConfig(d_model=16, d_visual=8, n_heads=2,
                                     n_visual_tokens=4)

    def test_perceiver_shape(self):
        cfg  = self._cfg()
        pr   = PerceiverResampler(cfg)
        vis  = torch.randn(2, 17, 8)
        out  = pr(vis)
        assert out.shape == (2, 4, 8)   # n_visual_tokens=4

    def test_visual_projector_shape(self):
        proj = VisualProjector(d_visual=8, d_model=16, n_layers=2)
        vis  = torch.randn(2, 17, 8)
        out  = proj(vis)
        assert out.shape == (2, 17, 16)

    def test_cross_modal_attention_shape(self):
        cfg    = self._cfg()
        xattn  = CrossModalAttention(cfg)
        text_h = torch.randn(2, 6, 16)
        vis    = torch.randn(2, 4, 8)
        out    = xattn(text_h, vis)
        assert out.shape == text_h.shape

    def test_cross_modal_gate_zero_at_init(self):
        cfg   = self._cfg()
        xattn = CrossModalAttention(cfg)
        assert abs(xattn.gate.item()) < 0.01   # initialized to zero


# ── VQA ───────────────────────────────────────────────────────────────────────

class TestVQA:
    def _models(self):
        enc = VisionEncoder(make_cfg())
        lm  = TinyLM()
        cfg = VQAConfig(d_visual=8, d_model=16, vocab_size=V)
        return enc, lm, cfg

    def test_early_fusion_shape(self):
        enc, lm, cfg = self._models()
        vqa  = EarlyFusionVQA(enc, lm, cfg)
        imgs = torch.randn(2, 3, 16, 16)
        q    = torch.randint(0, V, (2, 5))
        out  = vqa(imgs, q)
        assert out.dim() == 3 and out.shape[-1] == V

    def test_cross_attention_vqa_shape(self):
        enc, lm, cfg = self._models()
        vqa  = CrossAttentionVQA(enc, lm, cfg)
        imgs = torch.randn(2, 3, 16, 16)
        q    = torch.randint(0, V, (2, 5))
        out  = vqa(imgs, q)
        assert out.shape == (2, 5, V)

    def test_vqa_accuracy(self):
        assert vqa_accuracy(["cat", "2"], ["cat", "2"]) == 1.0
        assert vqa_accuracy(["cat", "dog"], ["cat", "2"]) == 0.5

    def test_generate_answer_list(self):
        enc, lm, cfg = self._models()
        vqa  = EarlyFusionVQA(enc, lm, cfg)
        imgs = torch.randn(1, 3, 16, 16)
        q    = torch.randint(0, V, (1, 4))
        ans  = vqa.generate_answer(imgs, q, max_len=3)
        assert isinstance(ans, list)


# ── Captioning ────────────────────────────────────────────────────────────────

class TestCaptioning:
    def _captioner(self):
        enc = VisionEncoder(make_cfg())
        lm  = TinyLM()
        cfg = CaptioningConfig(d_visual=8, d_model=16, vocab_size=V)
        return ImageCaptioner(enc, lm, cfg)

    def test_training_step_loss(self):
        c    = self._captioner()
        imgs = torch.randn(2, 3, 16, 16)
        tgt  = torch.randint(1, V-1, (2, 6))
        loss = c.training_step(imgs, tgt)
        assert loss.item() > 0

    def test_generate_returns_list(self):
        c    = self._captioner()
        imgs = torch.randn(1, 3, 16, 16)
        cap  = c.generate(imgs, max_new_tokens=5)
        assert isinstance(cap, list)
        assert cap[0] == 1   # BOS token

    def test_bleu_perfect(self):
        s = "the cat sat".split()
        assert abs(bleu_score(s, [s]) - 1.0) < 0.01

    def test_bleu_no_overlap(self):
        assert bleu_score("abc".split(), ["xyz".split()]) == 0.0

    def test_bleu_partial(self):
        c = "the cat".split()
        r = ["the cat sat on mat".split()]
        assert 0.0 < bleu_score(c, r) < 1.0


# ── Retrieval ──────────────────────────────────────────────────────────────────

class TestRetrieval:
    def _retriever(self):
        cfg = make_cfg()
        enc = VisionEncoder(cfg)
        lm  = TinyLM()
        clip = CLIPModel(enc, lm, d_text=16, cfg=CLIPConfig(d_embed=8))
        return CrossModalRetriever(clip)

    def test_index_and_i2t(self):
        r    = self._retriever()
        imgs = torch.randn(5, 3, 16, 16)
        txt  = torch.randint(0, V, (5, 6))
        r.index(imgs, txt)
        results = r.image_to_text(imgs[:1], k=3)
        assert len(results) == 3

    def test_t2i_results(self):
        r    = self._retriever()
        imgs = torch.randn(5, 3, 16, 16)
        txt  = torch.randint(0, V, (5, 6))
        r.index(imgs, txt)
        results = r.text_to_image(txt[:1], k=2)
        assert len(results) == 2

    def test_evaluate_returns_metrics(self):
        r    = self._retriever()
        imgs = torch.randn(4, 3, 16, 16)
        txt  = torch.randint(0, V, (4, 6))
        m    = r.evaluate(imgs, txt)
        assert isinstance(m, RetrievalMetrics)
        assert 0.0 <= m.r_at_1 <= 1.0

    def test_not_indexed_raises(self):
        r = self._retriever()
        with pytest.raises(RuntimeError):
            r.image_to_text(torch.randn(1, 3, 16, 16))
''')
commit("test: add full VLM test suite — ViT, CLIP, Perceiver, VQA, captioning, retrieval")

for title, body in [
    ("test: add VisionEncoderConfig seq_len test", '''
class TestVisionEncoderConfig:
    def test_seq_len(self):
        cfg = VisionEncoderConfig(image_size=16, patch_size=4)
        assert cfg.seq_len == cfg.n_patches + 1  # +1 for CLS

    def test_patch_dim(self):
        cfg = VisionEncoderConfig(image_size=16, patch_size=4, in_channels=3)
        assert cfg.patch_dim == 4 * 4 * 3
'''),
    ("test: add CLIPLoss temperature learnable test", '''
class TestCLIPLossTemp:
    def test_temperature_positive(self):
        loss = CLIPLoss(CLIPConfig(learn_temperature=True))
        assert loss.temperature.item() > 0

    def test_temperature_clipped(self):
        loss = CLIPLoss(CLIPConfig(temperature_max=10.0))
        assert loss.temperature.item() <= 10.0
'''),
    ("test: add PerceiverResampler latents shape test", '''
class TestPerceiverLatents:
    def test_learnable_latents(self):
        cfg = CrossAttentionConfig(d_visual=8, d_model=16, n_visual_tokens=6)
        pr  = PerceiverResampler(cfg)
        assert pr.latents.shape == (1, 6, 8)

    def test_output_fixed_length(self):
        cfg = CrossAttentionConfig(d_visual=8, d_model=16, n_visual_tokens=6)
        pr  = PerceiverResampler(cfg)
        # Different length inputs → same output length
        out1 = pr(torch.randn(2, 17, 8))
        out2 = pr(torch.randn(2, 49, 8))
        assert out1.shape == out2.shape
'''),
    ("test: add BLEU score max n-gram order test", '''
class TestBLEUOrders:
    def test_bleu1_higher_than_bleu4(self):
        from nanomind.vlm import bleu_score
        c = "cat sat mat".split()
        r = ["the cat sat on the mat".split()]
        b1 = bleu_score(c, r, max_n=1)
        b4 = bleu_score(c, r, max_n=4)
        assert b1 >= b4

    def test_empty_candidate(self):
        assert bleu_score([], [["cat sat".split()]]) == 0.0
'''),
    ("test: add VQASample to_dict test", '''
class TestVQASample:
    def test_to_dict(self):
        s = VQASample("img1", "What color?", "red", q_type="what")
        d = s.to_dict()
        assert d["answer"] == "red" and d["q_type"] == "what"
'''),
    ("test: add CrossModalRetriever score range test", '''
class TestRetrievalScores:
    def test_scores_in_range(self):
        enc  = VisionEncoder(make_cfg())
        lm   = TinyLM()
        clip = CLIPModel(enc, lm, d_text=16, cfg=CLIPConfig(d_embed=8))
        r    = CrossModalRetriever(clip)
        imgs = torch.randn(4, 3, 16, 16)
        txt  = torch.randint(0, V, (4, 6))
        r.index(imgs, txt)
        results = r.image_to_text(imgs[:1], k=4)
        for res in results:
            assert -1.0 <= res.score <= 1.0
'''),
    ("test: add ImageCaptioner BOS token test", '''
class TestCaptionerBOS:
    def test_starts_with_bos(self):
        enc = VisionEncoder(make_cfg())
        lm  = TinyLM()
        cfg = CaptioningConfig(d_visual=8, d_model=16, vocab_size=V, bos_token=1)
        c   = ImageCaptioner(enc, lm, cfg)
        cap = c.generate(torch.randn(1, 3, 16, 16), max_new_tokens=3)
        assert cap[0] == 1   # must start with BOS
'''),
]:
    src = read("tests/test_vlm.py")
    src += "\n" + body
    write("tests/test_vlm.py", src)
    commit(title)

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v5.5.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"5.4.0\"", "__version__ = \"5.5.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v5.5.0 — Vision-Language Models release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `eval`       | Evaluation Suite — MMLU/GSM8K/HellaSwag, metrics, LLM judge, leaderboard, Elo |",
    "| `eval`       | Evaluation Suite — MMLU/GSM8K/HellaSwag, metrics, LLM judge, leaderboard, Elo |\n"
    "| `vlm`        | Vision-Language Models — CLIP, ViT, VQA (LLaVA/Flamingo), captioning, retrieval |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [5.5.0] — 2024 — Vision-Language Models\n\n### Added\n"
      "- `VisionEncoder` — ViT patch embedding, transformer blocks, CLS embedding\n"
      "- `PatchEmbedding` — patchify 2D images + learnable positional embeddings\n"
      "- `CLIPLoss` — symmetric InfoNCE contrastive loss with learnable temperature\n"
      "- `CLIPModel` — encode_image, encode_text, zero_shot_classify\n"
      "- `PerceiverResampler` — Flamingo-style: compress N patches → K latent tokens\n"
      "- `VisualProjector` — LLaVA-style MLP: vision dim → LM dim\n"
      "- `CrossModalAttention` — text attends to visual tokens (gated, Flamingo)\n"
      "- `EarlyFusionVQA` — LLaVA-style: concatenate visual + text tokens\n"
      "- `CrossAttentionVQA` — Flamingo-style: cross-attention VQA\n"
      "- `vqa_accuracy` — VQA evaluation metric\n"
      "- `ImageCaptioner` — teacher-forcing training, greedy generation\n"
      "- `bleu_score` — BLEU-4 captioning metric (n-gram precision + brevity penalty)\n"
      "- `CrossModalRetriever` — image/text retrieval + Recall@1/5/10 evaluation\n"
      "- `RetrievalMetrics` — R@1, R@5, R@10, median rank\n"
      "- `examples/vlm_demo.py` — full VLM pipeline demo\n\n---\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v5.5.0, update README and CHANGELOG for Day 55 VLM")

print("\n=== Pushing Day 55 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")
run("git", "tag", "-a", "v5.5.0", "-m", "NanoMind v5.5.0 — Vision-Language Models", check=False)
r = run("git", "push", "origin", "v5.5.0", check=False)
print("Tag v5.5.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 55 COMPLETE — v5.5.0 TAGGED! ===")
