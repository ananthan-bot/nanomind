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
