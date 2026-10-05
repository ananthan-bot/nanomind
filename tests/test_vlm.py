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


class TestVisionEncoderConfig:
    def test_seq_len(self):
        cfg = VisionEncoderConfig(image_size=16, patch_size=4)
        assert cfg.seq_len == cfg.n_patches + 1  # +1 for CLS

    def test_patch_dim(self):
        cfg = VisionEncoderConfig(image_size=16, patch_size=4, in_channels=3)
        assert cfg.patch_dim == 4 * 4 * 3


class TestCLIPLossTemp:
    def test_temperature_positive(self):
        loss = CLIPLoss(CLIPConfig(learn_temperature=True))
        assert loss.temperature.item() > 0

    def test_temperature_clipped(self):
        loss = CLIPLoss(CLIPConfig(temperature_max=10.0))
        assert loss.temperature.item() <= 10.0


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
