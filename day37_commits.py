"""
day37_commits.py — 20 atomic commits for Day 37: Multi-Modal Support (Vision + Text).
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

print("\n=== DAY 37: Multi-Modal Support (Vision + Text) — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — multimodal package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/multimodal/__init__.py",
      '"""NanoMind Multimodal sub-package — vision + language fusion."""\n')
commit("feat: add nanomind/multimodal/ package skeleton for vision-language fusion")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — ModalityConfig
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/multimodal/config.py", '''\
"""
nanomind/multimodal/config.py — Multimodal model configuration.

## How Multimodal LLMs Work

Vision-Language Models (VLMs) extend text LLMs to understand images:

  1. Image Patching (ViT-style):
       Input image (H × W × 3) → divide into patches (P × P pixels each)
       → N = (H/P) × (W/P) patches
       → Linear projection → N "image tokens" of dimension D

  2. Fusion (Early vs Late):
       Early: concatenate image tokens with text tokens before attention
              → "Prefix" style: [IMG_1, IMG_2, ..., IMG_N, text tokens]
              Used by: LLaVA, InternVL, Qwen-VL

       Late:  separate vision and text encoders, cross-attention to fuse
              Used by: Flamingo, CogVLM

  3. Special tokens:
       <image>    → placeholder in text prompt
       <img_start> / <img_end> → delimit image tokens in sequence

Models implementing this:
  LLaVA (Haotian Liu, 2023):   CLIP vision + Vicuna LLM
  InstructBLIP (Salesforce):   Q-Former for image-text alignment
  GPT-4V (OpenAI, 2023):       Proprietary
  Gemini (Google, 2023):       Native multimodal
  Qwen-VL (Alibaba, 2023):    Compact open VLM

References:
  LLaVA: https://arxiv.org/abs/2304.08485
  Flamingo: https://arxiv.org/abs/2204.14198
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ModalityConfig:
    """
    Configuration for multimodal (vision-language) model.

    Attributes:
        image_size:     Input image size (square, pixels).
        patch_size:     Patch size (must divide image_size evenly).
        vision_dim:     Vision encoder output dimension.
        fusion:         Fusion strategy: ``"prefix"`` or ``"cross"``.
        n_visual_tokens: Number of visual tokens after projection.
        image_token_id: Special token ID for ``<image>`` placeholder.
        img_start_id:   Token ID for ``<img_start>`` marker.
        img_end_id:     Token ID for ``<img_end>`` marker.
        normalize_mean: Per-channel mean for image normalisation.
        normalize_std:  Per-channel std for image normalisation.
    """
    image_size:      int       = 224
    patch_size:      int       = 16
    vision_dim:      int       = 128
    fusion:          str       = "prefix"
    n_visual_tokens: int       = 49         # (224/16)^2 = 196 → projected to 49
    image_token_id:  int       = 0
    img_start_id:    int       = 1
    img_end_id:      int       = 2
    normalize_mean:  list      = None
    normalize_std:   list      = None

    def __post_init__(self) -> None:
        if self.normalize_mean is None:
            self.normalize_mean = [0.485, 0.456, 0.406]   # ImageNet
        if self.normalize_std is None:
            self.normalize_std  = [0.229, 0.224, 0.225]
        assert self.image_size  % self.patch_size == 0
        assert self.fusion      in ("prefix", "cross")
        assert self.patch_size  >= 4
        assert self.vision_dim  >= 8

    @property
    def n_patches(self) -> int:
        """Total number of patches per image."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        """Flattened patch dimension (C × P × P)."""
        return 3 * self.patch_size * self.patch_size
''')
commit("feat: add ModalityConfig — image_size, patch_size, vision_dim, fusion, n_patches, patch_dim")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — image patchifier
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/multimodal/patchify.py", '''\
"""
nanomind/multimodal/patchify.py — Image patching (ViT-style).

Splits an image into non-overlapping patches, the standard approach
used by Vision Transformers (ViT, DeiT, CLIP, etc.).

Pipeline:
  Image (H, W, C)                      → raw pixels
  Normalise (subtract mean, divide std) → float32 in ~[-2, 2]
  Reshape into patches (N, P*P*C)       → N = (H/P)*(W/P) patches
  Linear projection (N, D)              → visual tokens for the LLM

Reference:
  Dosovitskiy et al. (2020) "An Image is Worth 16x16 Words"
  https://arxiv.org/abs/2010.11929
"""

from __future__ import annotations
import torch
import torch.nn as nn


def patchify(
    image:      torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """
    Split a batch of images into flattened patches.

    Args:
        image:      ``(B, C, H, W)`` image tensor.
        patch_size: Patch side length in pixels.

    Returns:
        ``(B, N, C*P*P)`` patch tensor where N = (H/P)*(W/P).

    Example::

        img     = torch.randn(1, 3, 224, 224)
        patches = patchify(img, patch_size=16)
        # → (1, 196, 768)   (196 patches, each 3×16×16=768-dim)
    """
    B, C, H, W = image.shape
    P           = patch_size
    assert H % P == 0 and W % P == 0, \
        f"Image size ({H}×{W}) must be divisible by patch_size ({P})"

    # Reshape: B, C, H/P, P, W/P, P  →  B, N, C*P*P
    x = image.reshape(B, C, H // P, P, W // P, P)
    x = x.permute(0, 2, 4, 1, 3, 5)          # B, H/P, W/P, C, P, P
    x = x.contiguous().reshape(B, -1, C * P * P)
    return x


def unpatchify(
    patches:    torch.Tensor,
    patch_size: int,
    image_size: int,
    channels:   int = 3,
) -> torch.Tensor:
    """
    Reconstruct images from patches (inverse of patchify).

    Args:
        patches:    ``(B, N, C*P*P)`` patch tensor.
        patch_size: Patch side length.
        image_size: Output image side length.
        channels:   Number of image channels.

    Returns:
        ``(B, C, H, W)`` image tensor.
    """
    B   = patches.shape[0]
    P   = patch_size
    G   = image_size // P
    C   = channels
    x   = patches.reshape(B, G, G, C, P, P)
    x   = x.permute(0, 3, 1, 4, 2, 5)        # B, C, G, P, G, P
    return x.contiguous().reshape(B, C, image_size, image_size)


class ImageNormalizer(nn.Module):
    """
    Normalise image tensors using channel-wise mean and std.

    Args:
        mean: Per-channel mean (default: ImageNet mean).
        std:  Per-channel std (default: ImageNet std).
    """

    def __init__(
        self,
        mean: list[float] = None,
        std:  list[float] = None,
    ) -> None:
        super().__init__()
        mean = mean or [0.485, 0.456, 0.406]
        std  = std  or [0.229, 0.224, 0.225]
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``(B, 3, H, W)`` image tensor."""
        return (x - self.mean) / (self.std + 1e-7)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse normalisation."""
        return x * self.std + self.mean
''')
commit("feat: add patchify(), unpatchify(), ImageNormalizer — ViT-style image tokenisation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — vision encoder (patch embedder + transformer)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/multimodal/vision_encoder.py", '''\
"""
nanomind/multimodal/vision_encoder.py — Vision encoder: patches → visual tokens.

Implements a lightweight Vision Transformer (ViT) encoder that maps
image patches to visual token embeddings compatible with the LLM.

Architecture:
  Image → Normalise → Patchify → Linear(patch_dim, vision_dim)
       → Positional Embedding → Transformer Layers → Visual Tokens

This is the vision half of LLaVA / InstructBLIP / Qwen-VL:
  - LLaVA uses CLIP ViT-L/14 (307M params) as the vision encoder
  - NanoMind implements a miniature ViT for educational clarity
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.multimodal.config import ModalityConfig
from nanomind.multimodal.patchify import patchify, ImageNormalizer


class ViTBlock(nn.Module):
    """
    Single Vision Transformer block: LayerNorm → MHA → FFN.

    Args:
        d_model:  Hidden dimension.
        n_heads:  Number of attention heads.
        mlp_ratio: FFN expansion ratio (default: 4×).
        dropout:  Attention and FFN dropout.
    """

    def __init__(
        self,
        d_model:   int,
        n_heads:   int,
        mlp_ratio: float = 4.0,
        dropout:   float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                            dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        d_ff       = int(d_model * mlp_ratio)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        h  = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x  = x + h
        # Pre-norm FFN
        x  = x + self.ffn(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """
    Lightweight Vision Transformer encoder for NanoMind.

    Converts images to visual token sequences compatible with the LLM.

    Args:
        cfg:       Modality configuration.
        n_layers:  Number of ViT transformer layers.
        n_heads:   Number of attention heads in vision transformer.

    Example::

        encoder = VisionEncoder(ModalityConfig())
        image   = torch.randn(1, 3, 224, 224)   # batch of 1 image
        tokens  = encoder(image)
        # → (1, 196, vision_dim)  visual token sequence
    """

    def __init__(
        self,
        cfg:      ModalityConfig,
        n_layers: int = 2,
        n_heads:  int = 4,
    ) -> None:
        super().__init__()
        self.cfg        = cfg
        self.normalizer = ImageNormalizer(cfg.normalize_mean, cfg.normalize_std)
        self.patch_proj = nn.Linear(cfg.patch_dim, cfg.vision_dim)
        self.pos_embed  = nn.Parameter(
            torch.zeros(1, cfg.n_patches, cfg.vision_dim)
        )
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, cfg.vision_dim))
        self.layers     = nn.ModuleList([
            ViTBlock(cfg.vision_dim, n_heads) for _ in range(n_layers)
        ])
        self.norm       = nn.LayerNorm(cfg.vision_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode an image batch to visual tokens.

        Args:
            image: ``(B, 3, H, W)`` image tensor (float32, pixels in [0, 1]).

        Returns:
            ``(B, N, vision_dim)`` visual token tensor.
        """
        B = image.shape[0]
        x = self.normalizer(image)
        x = patchify(x, self.cfg.patch_size)          # (B, N, patch_dim)
        x = self.patch_proj(x)                         # (B, N, vision_dim)
        x = x + self.pos_embed                         # add positional embedding
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)               # (B, N+1, vision_dim)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x[:, 1:, :]   # remove CLS, return patch tokens (B, N, vision_dim)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
''')
commit("feat: add ViTBlock + VisionEncoder — patch projection, CLS token, pos_embed, n_params")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — visual projector (vision_dim → lm_dim)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/multimodal/projector.py", '''\
"""
nanomind/multimodal/projector.py — Visual projector: vision tokens → LLM space.

The projector bridges the vision encoder and the language model.
It maps visual features from the vision embedding space to the LLM
token embedding space so they can be concatenated with text tokens.

LLaVA uses a simple 2-layer MLP:
  vision_token (D_v) → Linear(D_v, D_lm) → GELU → Linear(D_lm, D_lm)

Qwen-VL and InternVL use more complex cross-attention Q-Formers.
NanoMind implements MLP and optional token compression (pooling).

Reference:
  LLaVA: Visual Instruction Tuning — https://arxiv.org/abs/2304.08485
  Q-Former (InstructBLIP): https://arxiv.org/abs/2305.06500
"""

from __future__ import annotations
import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """
    Two-layer MLP projector to map visual features to LLM space.

    Args:
        vision_dim:  Input visual token dimension.
        lm_dim:      Output LLM token dimension.
        hidden_dim:  Hidden layer dimension (default: max of both).

    Example::

        proj   = MLPProjector(vision_dim=128, lm_dim=256)
        visual = torch.randn(1, 196, 128)
        mapped = proj(visual)   # → (1, 196, 256)
    """

    def __init__(
        self,
        vision_dim: int,
        lm_dim:     int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or max(vision_dim, lm_dim)
        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, lm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``(B, N, vision_dim)`` → ``(B, N, lm_dim)``."""
        return self.net(x)


class PoolingProjector(nn.Module):
    """
    Pooling projector: reduce N visual tokens to M tokens via average pooling.

    Reduces sequence length before injecting into the LLM, saving compute.
    Used by efficient VLMs to reduce from 196 → 36 visual tokens.

    Args:
        vision_dim:   Input visual token dimension.
        lm_dim:       Output LLM token dimension.
        target_tokens: Number of output tokens (M < N).
    """

    def __init__(
        self,
        vision_dim:    int,
        lm_dim:        int,
        target_tokens: int = 36,
    ) -> None:
        super().__init__()
        self.target_tokens = target_tokens
        self.proj = nn.Linear(vision_dim, lm_dim)
        self.norm = nn.LayerNorm(lm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool and project ``(B, N, vision_dim)`` → ``(B, target_tokens, lm_dim)``.
        """
        B, N, D = x.shape
        if N > self.target_tokens:
            # Adaptive average pooling over the token dimension
            x = x.permute(0, 2, 1)                         # (B, D, N)
            x = nn.functional.adaptive_avg_pool1d(x, self.target_tokens)
            x = x.permute(0, 2, 1)                         # (B, T, D)
        return self.norm(self.proj(x))


import torch.nn.functional as F   # noqa: E402 (re-import for PoolingProjector)
''')
commit("feat: add MLPProjector (2-layer MLP), PoolingProjector (adaptive pool + project)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — multimodal inputs
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/multimodal/inputs.py", '''\
"""
nanomind/multimodal/inputs.py — Multimodal input types and processors.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class ImageInput:
    """
    A single image input for multimodal inference.

    Attributes:
        pixels:   Image tensor ``(C, H, W)`` in [0, 1] float32.
        metadata: Optional dict with source path, alt text, etc.
    """
    pixels:   torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        return tuple(self.pixels.shape)

    @classmethod
    def zeros(cls, size: int = 224) -> "ImageInput":
        """Create a blank (zero) image of given size."""
        return cls(pixels=torch.zeros(3, size, size))

    @classmethod
    def random(cls, size: int = 224) -> "ImageInput":
        """Create a random image for testing."""
        return cls(pixels=torch.rand(3, size, size))


@dataclass
class MultimodalInput:
    """
    Combined text + image input for a multimodal model.

    Attributes:
        text:   Text prompt (may contain ``<image>`` placeholder).
        images: List of :class:`ImageInput` objects.
        system: Optional system message.
    """
    text:   str
    images: list[ImageInput] = field(default_factory=list)
    system: str | None       = None

    @property
    def n_images(self) -> int:
        return len(self.images)

    def has_images(self) -> bool:
        return len(self.images) > 0

    def image_placeholder_count(self) -> int:
        """Count <image> placeholders in the text."""
        return self.text.count("<image>")


class MultimodalProcessor:
    """
    Process multimodal inputs: encode images and build token sequences.

    Args:
        encoder:   :class:`VisionEncoder` for image → visual tokens.
        projector: :class:`MLPProjector` for visual token → LLM space.
        tokenizer: Text tokenizer with encode/decode.
        cfg:       :class:`ModalityConfig`.
    """

    def __init__(self, encoder, projector, tokenizer, cfg) -> None:
        self.encoder   = encoder
        self.projector = projector
        self.tokenizer = tokenizer
        self.cfg       = cfg

    @torch.no_grad()
    def encode_images(self, images: list[ImageInput]) -> torch.Tensor:
        """
        Encode a list of images to projected visual tokens.

        Args:
            images: List of :class:`ImageInput`.

        Returns:
            ``(N_images, N_patches, lm_dim)`` visual token tensor.
        """
        pixels = torch.stack([img.pixels for img in images])  # (N, C, H, W)
        visual = self.encoder(pixels)                           # (N, N_patches, vision_dim)
        return self.projector(visual)                           # (N, N_patches, lm_dim)

    def build_token_sequence(
        self,
        mm_input:    "MultimodalInput",
        visual_embeds: torch.Tensor | None = None,
    ) -> dict:
        """
        Build the combined token sequence for multimodal input.

        Returns:
            Dict with ``input_ids``, ``image_positions``, ``n_visual_tokens``.
        """
        text_ids = self.tokenizer.encode(mm_input.text)
        return {
            "input_ids":       text_ids,
            "n_images":        mm_input.n_images,
            "n_visual_tokens": self.cfg.n_visual_tokens,
        }
''')
commit("feat: add ImageInput, MultimodalInput, MultimodalProcessor — encode_images, build_token_sequence")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — prefix fusion (prepend image tokens to text)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/multimodal/fusion.py", '''\
"""
nanomind/multimodal/fusion.py — Image-text fusion strategies.

Defines how visual tokens are merged with text token embeddings:

  Prefix Fusion (LLaVA-style):
    [<img_start>, v_1, v_2, ..., v_N, <img_end>, text tokens...]
    Simple concatenation — visual tokens prefix the text.
    The LLM sees N additional "virtual" tokens before the text.

  Cross-Attention Fusion (Flamingo-style):
    Text tokens attend to visual tokens via gated cross-attention.
    More parameter-efficient but architecturally more complex.
    Allows the LLM to selectively query visual information.

NanoMind implements prefix fusion (simpler, more widely used).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class PrefixFusion(nn.Module):
    """
    Prefix fusion: prepend projected visual tokens to text embeddings.

    Args:
        lm_embed_dim: LLM embedding dimension.

    Example::

        fusion  = PrefixFusion(lm_embed_dim=256)
        vis_tok = torch.randn(1, 49, 256)   # projected visual tokens
        txt_emb = torch.randn(1, 20, 256)   # text token embeddings
        fused   = fusion(vis_tok, txt_emb)
        # → (1, 69, 256)   visual prefix + text
    """

    def __init__(self, lm_embed_dim: int) -> None:
        super().__init__()
        self.lm_embed_dim = lm_embed_dim
        # Learnable scale for visual tokens (initialised to 1)
        self.visual_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        visual_embeds: torch.Tensor,
        text_embeds:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse visual prefix with text embeddings.

        Args:
            visual_embeds: ``(B, N_visual, D)`` projected visual tokens.
            text_embeds:   ``(B, N_text, D)`` text token embeddings.

        Returns:
            ``(B, N_visual + N_text, D)`` fused embedding sequence.
        """
        scaled_vis = visual_embeds * self.visual_scale
        return torch.cat([scaled_vis, text_embeds], dim=1)

    @property
    def output_len(self) -> str:
        return "N_visual + N_text"


class CrossAttentionFusion(nn.Module):
    """
    Gated cross-attention fusion (Flamingo-style).

    Text tokens attend to visual tokens via cross-attention.
    A tanh-gated residual controls the visual influence.

    Args:
        lm_embed_dim: LLM token dimension.
        n_heads:      Number of cross-attention heads.
    """

    def __init__(self, lm_embed_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            lm_embed_dim, n_heads, batch_first=True
        )
        self.norm  = nn.LayerNorm(lm_embed_dim)
        self.gate  = nn.Parameter(torch.zeros(1))   # tanh gate, starts at 0

    def forward(
        self,
        visual_embeds: torch.Tensor,
        text_embeds:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-attend text tokens to visual tokens.

        Args:
            visual_embeds: ``(B, N_visual, D)`` visual context.
            text_embeds:   ``(B, N_text, D)`` text queries.

        Returns:
            ``(B, N_text, D)`` text tokens with injected visual context.
        """
        normed  = self.norm(text_embeds)
        attn_out, _ = self.cross_attn(
            normed, visual_embeds, visual_embeds,
            need_weights=False,
        )
        # Gated residual: tanh(gate) starts at 0, learns to open gradually
        return text_embeds + torch.tanh(self.gate) * attn_out
''')
commit("feat: add PrefixFusion (visual_scale, cat), CrossAttentionFusion (gated tanh residual)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — multimodal model
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/multimodal/model.py", '''\
"""
nanomind/multimodal/model.py — Full vision-language model.

Combines:
  VisionEncoder → MLPProjector → PrefixFusion → NanoMind LLM
"""

from __future__ import annotations
import torch
import torch.nn as nn

from nanomind.multimodal.config import ModalityConfig
from nanomind.multimodal.vision_encoder import VisionEncoder
from nanomind.multimodal.projector import MLPProjector, PoolingProjector
from nanomind.multimodal.fusion import PrefixFusion, CrossAttentionFusion
from nanomind.multimodal.inputs import MultimodalInput, ImageInput
from nanomind.utils.logger import get_logger

log = get_logger("multimodal.model")


class VisionLanguageModel(nn.Module):
    """
    Vision-Language Model: encode images and generate text.

    Combines a VisionEncoder, a projector, and any text LLM backbone.

    Args:
        lm:          Language model backbone (must have ``lm.tok_emb``).
        cfg:         Modality configuration.
        lm_dim:      LLM embedding dimension.
        n_vis_layers: Number of ViT layers in the vision encoder.

    Example::

        vlm   = VisionLanguageModel(lm=nanomind_model, cfg=ModalityConfig(), lm_dim=128)
        image = ImageInput.random()
        out   = vlm.encode_image(image)   # → (1, N, lm_dim) visual tokens
    """

    def __init__(
        self,
        lm:           nn.Module,
        cfg:          ModalityConfig,
        lm_dim:       int,
        n_vis_layers: int = 2,
        n_vis_heads:  int = 4,
        use_pooling:  bool = True,
    ) -> None:
        super().__init__()
        self.cfg     = cfg
        self.lm      = lm
        self.encoder = VisionEncoder(cfg, n_layers=n_vis_layers, n_heads=n_vis_heads)
        self.projector = (
            PoolingProjector(cfg.vision_dim, lm_dim, target_tokens=cfg.n_visual_tokens)
            if use_pooling
            else MLPProjector(cfg.vision_dim, lm_dim)
        )
        self.fusion  = (
            PrefixFusion(lm_dim)
            if cfg.fusion == "prefix"
            else CrossAttentionFusion(lm_dim)
        )

    @torch.no_grad()
    def encode_image(self, image: "ImageInput") -> torch.Tensor:
        """
        Encode a single image to projected visual tokens.

        Args:
            image: :class:`ImageInput`.

        Returns:
            ``(1, N_visual, lm_dim)`` tensor.
        """
        pixels  = image.pixels.unsqueeze(0)               # (1, C, H, W)
        visual  = self.encoder(pixels)                     # (1, N_patch, vision_dim)
        return self.projector(visual)                      # (1, N_visual, lm_dim)

    def visual_token_count(self) -> int:
        """Number of visual tokens produced per image."""
        return self.cfg.n_visual_tokens

    @property
    def n_params(self) -> dict:
        """Parameter counts by component."""
        return {
            "vision_encoder": self.encoder.n_params,
            "projector":      sum(p.numel() for p in self.projector.parameters()),
            "fusion":         sum(p.numel() for p in self.fusion.parameters()),
            "lm":             sum(p.numel() for p in self.lm.parameters()),
            "total":          sum(p.numel() for p in self.parameters()),
        }
''')
commit("feat: add VisionLanguageModel — VisionEncoder + projector + fusion + LM, encode_image, n_params")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — image data augmentation
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/multimodal/augment.py", '''\
"""
nanomind/multimodal/augment.py — Image augmentation for vision-language training.

Augmentation is critical for robust visual grounding:
  - Random crop + resize: model learns scale invariance
  - Color jitter: model learns color-invariant features
  - Random flip:  model learns spatial invariance (for non-directional tasks)

Note: Some VQA tasks are directional (left/right), so horizontal flipping
must be disabled for position-sensitive tasks.

References:
  SimCLR augmentation: https://arxiv.org/abs/2002.05709
  CLIP training: https://arxiv.org/abs/2103.00020
"""

from __future__ import annotations
import random
import torch
import torch.nn.functional as F


class RandomCropResize:
    """Random crop and resize to target size."""

    def __init__(self, size: int = 224, scale: tuple = (0.8, 1.0)) -> None:
        self.size  = size
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply to ``(C, H, W)`` tensor."""
        C, H, W = x.shape
        scale   = random.uniform(*self.scale)
        new_h   = int(H * scale)
        new_w   = int(W * scale)
        top     = random.randint(0, H - new_h)
        left    = random.randint(0, W - new_w)
        x       = x[:, top:top + new_h, left:left + new_w]
        x       = F.interpolate(x.unsqueeze(0), size=(self.size, self.size),
                                mode="bilinear", align_corners=False).squeeze(0)
        return x


class RandomHorizontalFlip:
    """Randomly flip image horizontally with given probability."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return x.flip(-1)
        return x


class ColorJitter:
    """Random brightness, contrast, saturation, hue jitter."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast:   float = 0.2,
        saturation: float = 0.1,
    ) -> None:
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply to ``(C, H, W)`` tensor in [0, 1]."""
        # Brightness
        if self.brightness > 0:
            f = 1.0 + random.uniform(-self.brightness, self.brightness)
            x = (x * f).clamp(0.0, 1.0)
        # Contrast
        if self.contrast > 0:
            mean = x.mean(dim=(-2, -1), keepdim=True)
            f    = 1.0 + random.uniform(-self.contrast, self.contrast)
            x    = ((x - mean) * f + mean).clamp(0.0, 1.0)
        return x


class ImageAugmentor:
    """
    Compose image augmentations for vision-language training.

    Args:
        image_size:  Target image size after augmentation.
        flip:        Include random horizontal flip.
        crop:        Include random crop-resize.
        color:       Include color jitter.

    Example::

        aug   = ImageAugmentor(image_size=224)
        pixel = torch.rand(3, 256, 256)
        out   = aug(pixel)   # → (3, 224, 224)
    """

    def __init__(
        self,
        image_size: int   = 224,
        flip:       bool  = True,
        crop:       bool  = True,
        color:      bool  = True,
    ) -> None:
        self.transforms = []
        if crop:
            self.transforms.append(RandomCropResize(image_size))
        if flip:
            self.transforms.append(RandomHorizontalFlip())
        if color:
            self.transforms.append(ColorJitter())
        self._resize = image_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        # Ensure correct output size
        if x.shape[-1] != self._resize or x.shape[-2] != self._resize:
            x = F.interpolate(x.unsqueeze(0), size=(self._resize, self._resize),
                              mode="bilinear", align_corners=False).squeeze(0)
        return x
''')
commit("feat: add RandomCropResize, RandomHorizontalFlip, ColorJitter, ImageAugmentor pipeline")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — multimodal __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/multimodal/__init__.py", '''\
"""NanoMind Multimodal sub-package — Vision + Language fusion.

Implements the full vision-language stack:
  Image → Patches → VisionEncoder → Projector → PrefixFusion → LLM

Follows the LLaVA / InternVL architecture:
  - Patchify images into ViT-style tokens
  - Encode with a lightweight Vision Transformer
  - Project to LLM embedding space via 2-layer MLP
  - Prepend as visual prefix tokens to text input

Primary exports:
    - :class:`ModalityConfig`         — image_size, patch_size, vision_dim, fusion
    - :class:`VisionEncoder`          — ViT-style image encoder
    - :class:`ViTBlock`               — single ViT transformer block
    - :class:`MLPProjector`           — 2-layer MLP vision→LLM projection
    - :class:`PoolingProjector`       — adaptive pool + project
    - :class:`PrefixFusion`           — visual token prefix concatenation
    - :class:`CrossAttentionFusion`   — gated cross-attention (Flamingo-style)
    - :class:`VisionLanguageModel`    — full VLM: encode_image(), n_params
    - :class:`ImageInput`             — single image wrapper
    - :class:`MultimodalInput`        — text + images combined input
    - :class:`MultimodalProcessor`    — encode_images(), build_token_sequence()
    - :func:`patchify`                — image → patch tensor
    - :func:`unpatchify`              — patch tensor → image
    - :class:`ImageNormalizer`        — channel mean/std normalisation
    - :class:`ImageAugmentor`         — crop/flip/color augmentation pipeline
"""

from nanomind.multimodal.config import ModalityConfig
from nanomind.multimodal.patchify import patchify, unpatchify, ImageNormalizer
from nanomind.multimodal.vision_encoder import VisionEncoder, ViTBlock
from nanomind.multimodal.projector import MLPProjector, PoolingProjector
from nanomind.multimodal.fusion import PrefixFusion, CrossAttentionFusion
from nanomind.multimodal.model import VisionLanguageModel
from nanomind.multimodal.inputs import ImageInput, MultimodalInput, MultimodalProcessor
from nanomind.multimodal.augment import (
    ImageAugmentor, RandomCropResize, RandomHorizontalFlip, ColorJitter
)

__all__ = [
    "ModalityConfig",
    "patchify", "unpatchify", "ImageNormalizer",
    "VisionEncoder", "ViTBlock",
    "MLPProjector", "PoolingProjector",
    "PrefixFusion", "CrossAttentionFusion",
    "VisionLanguageModel",
    "ImageInput", "MultimodalInput", "MultimodalProcessor",
    "ImageAugmentor", "RandomCropResize", "RandomHorizontalFlip", "ColorJitter",
]
''')
commit("refactor: export all multimodal components from nanomind/multimodal/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — example: multimodal_demo.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/multimodal_demo.py", '''\
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
print(f"\nModalityConfig:")
print(f"  image={cfg.image_size}×{cfg.image_size}, patch={cfg.patch_size}×{cfg.patch_size}")
print(f"  n_patches={cfg.n_patches}, patch_dim={cfg.patch_dim}")

# ── Patchify ──────────────────────────────────────────────────────────────────
image  = torch.rand(1, 3, 32, 32)
patches = patchify(image, patch_size=8)
print(f"\nPatchify: {tuple(image.shape)} → {tuple(patches.shape)}")
recon  = unpatchify(patches, patch_size=8, image_size=32)
print(f"Unpatchify: {tuple(recon.shape)} | match={torch.allclose(image, recon)}")

# ── ImageNormalizer ───────────────────────────────────────────────────────────
norm = ImageNormalizer()
normalised = norm(image)
print(f"\nNormalized range: [{normalised.min():.3f}, {normalised.max():.3f}]")

# ── VisionEncoder ─────────────────────────────────────────────────────────────
encoder = VisionEncoder(cfg, n_layers=2, n_heads=4)
visual  = encoder(image)
print(f"\nVisionEncoder: {tuple(image.shape)} → {tuple(visual.shape)}")
print(f"  n_params: {encoder.n_params:,}")

# ── Projectors ────────────────────────────────────────────────────────────────
mlp_proj  = MLPProjector(vision_dim=32, lm_dim=LM_DIM)
pool_proj = PoolingProjector(vision_dim=32, lm_dim=LM_DIM, target_tokens=8)
mlp_out   = mlp_proj(visual)
pool_out  = pool_proj(visual)
print(f"\nMLPProjector:     {tuple(visual.shape)} → {tuple(mlp_out.shape)}")
print(f"PoolingProjector: {tuple(visual.shape)} → {tuple(pool_out.shape)}")

# ── Fusion ────────────────────────────────────────────────────────────────────
prefix_fusion = PrefixFusion(lm_embed_dim=LM_DIM)
text_emb      = torch.randn(1, 10, LM_DIM)
fused         = prefix_fusion(pool_out, text_emb)
print(f"\nPrefixFusion: vis={tuple(pool_out.shape)} + txt={tuple(text_emb.shape)} → {tuple(fused.shape)}")

cross_fusion  = CrossAttentionFusion(lm_embed_dim=LM_DIM, n_heads=4)
cross_out     = cross_fusion(pool_out, text_emb)
print(f"CrossAttnFusion: → {tuple(cross_out.shape)}")

# ── VisionLanguageModel ───────────────────────────────────────────────────────
lm  = TinyLM()
vlm = VisionLanguageModel(lm, cfg, lm_dim=LM_DIM)
img = ImageInput.random(size=32)
vis_tokens = vlm.encode_image(img)
print(f"\nVisionLanguageModel.encode_image: → {tuple(vis_tokens.shape)}")
print(f"  n_params: {vlm.n_params}")

# ── Augmentation ──────────────────────────────────────────────────────────────
aug   = ImageAugmentor(image_size=32, flip=True, crop=True, color=True)
pixel = torch.rand(3, 32, 32)
out   = aug(pixel)
print(f"\nImageAugmentor: {tuple(pixel.shape)} → {tuple(out.shape)}")
print("\nMultimodal demo complete!")
''')
commit("feat: add examples/multimodal_demo.py — patchify, VisionEncoder, projectors, fusion, VLM, augment")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 12-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_multimodal.py", '''\
"""tests/test_multimodal.py — Tests for NanoMind multimodal (vision-language)."""
import pytest
import torch
import torch.nn as nn

from nanomind.multimodal import (
    ModalityConfig, patchify, unpatchify, ImageNormalizer,
    VisionEncoder, ViTBlock,
    MLPProjector, PoolingProjector,
    PrefixFusion, CrossAttentionFusion,
    VisionLanguageModel,
    ImageInput, MultimodalInput, MultimodalProcessor,
    ImageAugmentor, RandomCropResize, RandomHorizontalFlip, ColorJitter,
)

IMG_SIZE   = 32
PATCH_SIZE = 8
VISION_DIM = 16
LM_DIM     = 32
N_PATCHES  = (IMG_SIZE // PATCH_SIZE) ** 2   # 16

def tiny_cfg(**kw):
    return ModalityConfig(image_size=IMG_SIZE, patch_size=PATCH_SIZE,
                          vision_dim=VISION_DIM, n_visual_tokens=8, **kw)

def rand_image(B=1):
    return torch.rand(B, 3, IMG_SIZE, IMG_SIZE)

class TinyLM(nn.Module):
    def __init__(self, V=16, D=LM_DIM, T=32):
        super().__init__()
        self.T = T
        self.tok_emb = nn.Embedding(V, D)
        self.lm_head = nn.Linear(D, V)
    def forward(self, x, t=None):
        return self.lm_head(self.tok_emb(x)), None


# ── ModalityConfig ────────────────────────────────────────────────────────────

class TestModalityConfig:
    def test_n_patches(self):
        cfg = tiny_cfg()
        assert cfg.n_patches == N_PATCHES

    def test_patch_dim(self):
        cfg = tiny_cfg()
        assert cfg.patch_dim == 3 * PATCH_SIZE * PATCH_SIZE

    def test_invalid_patch_size(self):
        with pytest.raises(AssertionError):
            ModalityConfig(image_size=32, patch_size=7)  # 32 % 7 != 0

    def test_invalid_fusion(self):
        with pytest.raises(AssertionError):
            ModalityConfig(image_size=32, patch_size=8, fusion="early")

    def test_defaults(self):
        cfg = ModalityConfig()
        assert cfg.normalize_mean is not None
        assert len(cfg.normalize_mean) == 3


# ── Patchify ──────────────────────────────────────────────────────────────────

class TestPatchify:
    def test_shape(self):
        img     = rand_image()
        patches = patchify(img, PATCH_SIZE)
        assert patches.shape == (1, N_PATCHES, 3 * PATCH_SIZE ** 2)

    def test_batch(self):
        img     = rand_image(B=4)
        patches = patchify(img, PATCH_SIZE)
        assert patches.shape[0] == 4

    def test_roundtrip(self):
        img     = rand_image()
        patches = patchify(img, PATCH_SIZE)
        recon   = unpatchify(patches, PATCH_SIZE, IMG_SIZE)
        assert torch.allclose(img, recon, atol=1e-5)

    def test_invalid_patch_size_raises(self):
        img = rand_image()
        with pytest.raises(AssertionError):
            patchify(img, 7)  # 32 % 7 != 0


# ── ImageNormalizer ───────────────────────────────────────────────────────────

class TestImageNormalizer:
    def test_output_shape(self):
        norm = ImageNormalizer()
        img  = rand_image()
        out  = norm(img)
        assert out.shape == img.shape

    def test_changes_values(self):
        norm = ImageNormalizer()
        img  = rand_image()
        out  = norm(img)
        assert not torch.allclose(img, out)

    def test_denormalize_roundtrip(self):
        norm = ImageNormalizer()
        img  = rand_image()
        out  = norm.denormalize(norm(img))
        assert torch.allclose(img, out, atol=1e-5)


# ── ViTBlock ──────────────────────────────────────────────────────────────────

class TestViTBlock:
    def test_output_shape(self):
        block = ViTBlock(d_model=16, n_heads=2)
        x     = torch.randn(1, 8, 16)
        out   = block(x)
        assert out.shape == x.shape

    def test_batch_preserved(self):
        block = ViTBlock(d_model=16, n_heads=2)
        x     = torch.randn(3, 8, 16)
        out   = block(x)
        assert out.shape[0] == 3


# ── VisionEncoder ─────────────────────────────────────────────────────────────

class TestVisionEncoder:
    def _enc(self):
        return VisionEncoder(tiny_cfg(), n_layers=1, n_heads=2)

    def test_output_shape(self):
        enc = self._enc()
        out = enc(rand_image())
        assert out.shape == (1, N_PATCHES, VISION_DIM)

    def test_batch(self):
        enc = self._enc()
        out = enc(rand_image(B=3))
        assert out.shape[0] == 3

    def test_n_params_positive(self):
        enc = self._enc()
        assert enc.n_params > 0

    def test_gradient_flows(self):
        enc = self._enc()
        img = rand_image()
        img.requires_grad_(False)
        out = enc(img)
        loss = out.sum()
        loss.backward()
        for p in enc.parameters():
            if p.grad is not None:
                break
        else:
            pytest.fail("No gradients computed")


# ── Projectors ────────────────────────────────────────────────────────────────

class TestProjectors:
    def _visual(self, N=N_PATCHES):
        return torch.randn(1, N, VISION_DIM)

    def test_mlp_output_shape(self):
        proj = MLPProjector(VISION_DIM, LM_DIM)
        out  = proj(self._visual())
        assert out.shape == (1, N_PATCHES, LM_DIM)

    def test_pooling_reduces_tokens(self):
        proj = PoolingProjector(VISION_DIM, LM_DIM, target_tokens=8)
        out  = proj(self._visual())
        assert out.shape == (1, 8, LM_DIM)

    def test_mlp_batch(self):
        proj = MLPProjector(VISION_DIM, LM_DIM)
        vis  = torch.randn(4, N_PATCHES, VISION_DIM)
        out  = proj(vis)
        assert out.shape[0] == 4

    def test_pooling_already_small(self):
        """If N <= target, no pooling should be needed."""
        proj = PoolingProjector(VISION_DIM, LM_DIM, target_tokens=100)
        vis  = torch.randn(1, 8, VISION_DIM)
        out  = proj(vis)
        assert out.shape[1] == 8


# ── Fusion ────────────────────────────────────────────────────────────────────

class TestFusion:
    def _inputs(self, n_vis=8, n_txt=10):
        vis = torch.randn(1, n_vis, LM_DIM)
        txt = torch.randn(1, n_txt, LM_DIM)
        return vis, txt

    def test_prefix_fusion_shape(self):
        f    = PrefixFusion(LM_DIM)
        vis, txt = self._inputs()
        out  = f(vis, txt)
        assert out.shape == (1, 8 + 10, LM_DIM)

    def test_prefix_visual_scale_learnable(self):
        f = PrefixFusion(LM_DIM)
        assert f.visual_scale.requires_grad

    def test_cross_attn_fusion_shape(self):
        f    = CrossAttentionFusion(LM_DIM, n_heads=4)
        vis, txt = self._inputs()
        out  = f(vis, txt)
        assert out.shape == txt.shape   # text shape preserved

    def test_cross_attn_gate_zero_init(self):
        f = CrossAttentionFusion(LM_DIM)
        assert f.gate.item() == 0.0


# ── ImageInput / MultimodalInput ──────────────────────────────────────────────

class TestInputTypes:
    def test_image_input_zeros(self):
        img = ImageInput.zeros(32)
        assert img.pixels.sum() == 0.0

    def test_image_input_random(self):
        img = ImageInput.random(32)
        assert img.pixels.shape == (3, 32, 32)

    def test_image_input_shape(self):
        img = ImageInput.random(64)
        assert img.shape == (3, 64, 64)

    def test_multimodal_n_images(self):
        mm = MultimodalInput("hello <image> world",
                              images=[ImageInput.random(32), ImageInput.random(32)])
        assert mm.n_images == 2

    def test_multimodal_placeholder_count(self):
        mm = MultimodalInput("<image> and <image>")
        assert mm.image_placeholder_count() == 2

    def test_multimodal_has_images(self):
        mm = MultimodalInput("text only")
        assert not mm.has_images()


# ── VisionLanguageModel ───────────────────────────────────────────────────────

class TestVisionLanguageModel:
    def _vlm(self):
        lm  = TinyLM()
        cfg = tiny_cfg()
        return VisionLanguageModel(lm, cfg, lm_dim=LM_DIM, n_vis_layers=1, n_vis_heads=2)

    def test_encode_image_shape(self):
        vlm = self._vlm()
        img = ImageInput.random(IMG_SIZE)
        out = vlm.encode_image(img)
        assert out.shape[0] == 1
        assert out.shape[2] == LM_DIM

    def test_n_params_dict(self):
        vlm    = self._vlm()
        params = vlm.n_params
        for k in ("vision_encoder", "projector", "lm", "total"):
            assert k in params

    def test_visual_token_count(self):
        vlm = self._vlm()
        assert vlm.visual_token_count() == 8


# ── Augmentation ──────────────────────────────────────────────────────────────

class TestAugmentation:
    def _px(self):
        return torch.rand(3, 32, 32)

    def test_augmentor_output_shape(self):
        aug = ImageAugmentor(image_size=32)
        out = aug(self._px())
        assert out.shape == (3, 32, 32)

    def test_flip_changes_image(self):
        flip = RandomHorizontalFlip(p=1.0)  # always flip
        x    = torch.rand(3, 8, 8)
        out  = flip(x)
        assert not torch.allclose(x, out)

    def test_flip_twice_is_identity(self):
        flip = RandomHorizontalFlip(p=1.0)
        x    = torch.rand(3, 8, 8)
        assert torch.allclose(flip(flip(x)), x)

    def test_color_jitter_clamps(self):
        jitter = ColorJitter(brightness=0.9)
        x      = torch.rand(3, 8, 8)
        out    = jitter(x)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_crop_resize_output_size(self):
        crop = RandomCropResize(size=32, scale=(0.5, 1.0))
        x    = torch.rand(3, 32, 32)
        out  = crop(x)
        assert out.shape == (3, 32, 32)
''')
commit("test: add full multimodal test suite — config, patchify, normalizer, ViTBlock, encoder, projectors, fusion, inputs, VLM, augment")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 13-18 (additional test commits)
# ══════════════════════════════════════════════════════════════════════════════

# COMMIT 13 — patchify batch test
src = read("tests/test_multimodal.py")
src += '''

# ── Patchify batch consistency ────────────────────────────────────────────────

class TestPatchifyBatch:
    def test_each_image_independent(self):
        img1    = torch.ones(1, 3, 32, 32)
        img2    = torch.zeros(1, 3, 32, 32)
        batch   = torch.cat([img1, img2], dim=0)
        patches = patchify(batch, PATCH_SIZE)
        assert torch.allclose(patches[0], torch.ones_like(patches[0]))
        assert torch.allclose(patches[1], torch.zeros_like(patches[1]))

    def test_different_patch_sizes(self):
        for ps in [4, 8, 16]:
            img = torch.rand(1, 3, 32, 32) if 32 % ps == 0 else torch.rand(1, 3, 16, 16)
            s   = 32 if ps <= 8 else 16
            if s % ps != 0:
                continue
            patches = patchify(torch.rand(1, 3, s, s), ps)
            assert patches.shape[1] == (s // ps) ** 2
'''
write("tests/test_multimodal.py", src)
commit("test: add patchify batch independence and multiple patch_size tests")

# COMMIT 14 — VisionEncoder normalizer integration
src = read("tests/test_multimodal.py")
src += '''

# ── VisionEncoder with normalizer ─────────────────────────────────────────────

class TestVisionEncoderNormalizer:
    def test_normalizer_inside_encoder(self):
        enc = VisionEncoder(tiny_cfg(), n_layers=1, n_heads=2)
        # Encoder internally normalizes
        img1 = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        img2 = torch.ones(1, 3, IMG_SIZE, IMG_SIZE)
        out1 = enc(img1)
        out2 = enc(img2)
        # Different inputs should produce different outputs
        assert not torch.allclose(out1, out2)

    def test_encoder_eval_no_grad(self):
        enc = VisionEncoder(tiny_cfg(), n_layers=1, n_heads=2)
        enc.eval()
        with torch.no_grad():
            out = enc(rand_image())
        assert out.shape[-1] == VISION_DIM
'''
write("tests/test_multimodal.py", src)
commit("test: add VisionEncoder normalizer integration and eval no_grad tests")

# COMMIT 15 — MLPProjector hidden dim
src = read("tests/test_multimodal.py")
src += '''

# ── MLPProjector hidden dim ───────────────────────────────────────────────────

class TestMLPProjectorHiddenDim:
    def test_custom_hidden_dim(self):
        proj = MLPProjector(VISION_DIM, LM_DIM, hidden_dim=64)
        vis  = torch.randn(1, 8, VISION_DIM)
        out  = proj(vis)
        assert out.shape == (1, 8, LM_DIM)

    def test_gradient_through_projector(self):
        proj = MLPProjector(VISION_DIM, LM_DIM)
        vis  = torch.randn(1, 8, VISION_DIM, requires_grad=True)
        out  = proj(vis).sum()
        out.backward()
        assert vis.grad is not None
'''
write("tests/test_multimodal.py", src)
commit("test: add MLPProjector custom hidden_dim and gradient flow tests")

# COMMIT 16 — CrossAttentionFusion gate learns
src = read("tests/test_multimodal.py")
src += '''

# ── CrossAttentionFusion gate ─────────────────────────────────────────────────

class TestCrossAttnGate:
    def test_gate_gradient(self):
        f        = CrossAttentionFusion(LM_DIM, n_heads=4)
        vis, txt = torch.randn(1, 8, LM_DIM), torch.randn(1, 6, LM_DIM)
        out      = f(vis, txt).sum()
        out.backward()
        assert f.gate.grad is not None

    def test_gate_zero_means_no_visual_influence(self):
        """With gate=0, tanh(0)=0, so output should equal input."""
        f        = CrossAttentionFusion(LM_DIM, n_heads=4)
        f.gate.data.fill_(0.0)
        vis, txt = torch.randn(1, 8, LM_DIM), torch.randn(1, 6, LM_DIM)
        with torch.no_grad():
            out = f(vis, txt)
        assert torch.allclose(out, txt, atol=1e-5)
'''
write("tests/test_multimodal.py", src)
commit("test: add CrossAttentionFusion gate gradient and zero-gate no-visual-influence tests")

# COMMIT 17 — augmentation determinism at p=0
src = read("tests/test_multimodal.py")
src += '''

# ── Augmentation determinism ──────────────────────────────────────────────────

class TestAugmentDeterminism:
    def test_no_flip_p0_identity(self):
        flip = RandomHorizontalFlip(p=0.0)
        x    = torch.rand(3, 8, 8)
        assert torch.allclose(flip(x), x)

    def test_augmentor_no_augmentations(self):
        aug = ImageAugmentor(image_size=32, flip=False, crop=False, color=False)
        x   = torch.rand(3, 32, 32)
        out = aug(x)
        assert out.shape == (3, 32, 32)

    def test_color_zero_jitter_approx_identity(self):
        jitter = ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0)
        x      = torch.rand(3, 8, 8)
        out    = jitter(x)
        assert torch.allclose(x, out, atol=1e-5)
'''
write("tests/test_multimodal.py", src)
commit("test: add augmentation determinism — p=0 no-flip, no-augmentation, zero-jitter identity tests")

# COMMIT 18 — ImageInput metadata
src = read("tests/test_multimodal.py")
src += '''

# ── ImageInput metadata ───────────────────────────────────────────────────────

class TestImageInputMetadata:
    def test_metadata_stored(self):
        img = ImageInput(pixels=torch.rand(3, 32, 32),
                          metadata={"source": "test.jpg", "label": "cat"})
        assert img.metadata["label"] == "cat"

    def test_multimodal_system_prompt(self):
        mm = MultimodalInput("text", system="You are a vision assistant.")
        assert mm.system == "You are a vision assistant."

    def test_multimodal_no_images(self):
        mm = MultimodalInput("text only")
        assert mm.n_images == 0
        assert not mm.has_images()
'''
write("tests/test_multimodal.py", src)
commit("test: add ImageInput metadata stored, MultimodalInput system_prompt, no-images tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v3.7.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"3.6.0\"", "__version__ = \"3.7.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v3.7.0 — Multi-Modal Support (Vision + Language) release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `cache`     | KV-Cache — LayerCache, CacheManager, PrefixCache, SpeculativeDecoder |",
    "| `cache`     | KV-Cache — LayerCache, CacheManager, PrefixCache, SpeculativeDecoder |\n"
    "| `multimodal` | Vision-Language — VisionEncoder, projectors, fusion, VLM, augmentation |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = "## [3.7.0] — 2024 — Multi-Modal Support (Vision + Language)\n\n### Added\n" \
     "- `ModalityConfig` — image_size, patch_size, vision_dim, fusion, n_patches\n" \
     "- `patchify()` / `unpatchify()` — ViT-style image tokenisation\n" \
     "- `ImageNormalizer` — channel mean/std normalisation with denormalize()\n" \
     "- `ViTBlock` — LayerNorm + MHA + FFN vision transformer block\n" \
     "- `VisionEncoder` — CLS token, positional embedding, n_params\n" \
     "- `MLPProjector` — 2-layer MLP vision→LLM projection\n" \
     "- `PoolingProjector` — adaptive avg pool + linear projection\n" \
     "- `PrefixFusion` — learnable visual_scale + prefix concatenation\n" \
     "- `CrossAttentionFusion` — gated tanh cross-attention (Flamingo-style)\n" \
     "- `VisionLanguageModel` — encode_image(), n_params dict\n" \
     "- `ImageInput` / `MultimodalInput` — input types with metadata\n" \
     "- `ImageAugmentor` — RandomCropResize + RandomHorizontalFlip + ColorJitter\n" \
     "- `examples/multimodal_demo.py` — full VLM pipeline demo\n\n---\n\n" + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v3.7.0, update README and CHANGELOG for Day 37 Multi-Modal")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 37 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v3.7.0",
    "-m", "NanoMind v3.7.0 — Multi-Modal Support (Vision + Language)", check=False)
r = run("git", "push", "origin", "v3.7.0", check=False)
print("Tag v3.7.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 37 COMPLETE — v3.7.0 TAGGED! ===")
