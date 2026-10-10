"""
tests/test_omni.py — Unit tests for NanoMind-Omni unified architecture.
"""
import torch
from nanomind.omni import (
    ModalityType,
    OmniConfig,
    MultimodalSequenceBuilder,
    ModalityEmbedding,
    CrossModalFusionLayer,
    DuplexDialogueManager,
    DialogueState,
    VoiceActivityDetector,
    EndOfUtterancePredictor,
    OmniGenerator,
    NanoMindOmni,
    OmniMultiTaskLoss,
    OmniBenchmark,
    OmniPipeline,
)


class TestInterleaveAndFusion:
    def test_multimodal_interleaving(self):
        builder = MultimodalSequenceBuilder(d_model=32)
        t1 = torch.randn(2, 4, 32)
        t2 = torch.randn(2, 3, 32)
        seq, ids = builder.build_sequence([
            (ModalityType.TEXT, t1),
            (ModalityType.IMAGE, t2),
        ])
        assert seq.shape == (2, 7, 32)
        assert ids.shape == (2, 7)
        assert (ids[:, :4] == 0).all()
        assert (ids[:, 4:] == 1).all()

    def test_modality_embedding(self):
        mod_emb = ModalityEmbedding(num_modalities=4, d_model=32)
        ids = torch.tensor([[0, 1, 2]])
        emb = mod_emb(ids)
        assert emb.shape == (1, 3, 32)

    def test_fusion_layer(self):
        layer = CrossModalFusionLayer(d_model=32, n_heads=2)
        x = torch.randn(2, 6, 32)
        out = layer(x)
        assert out.shape == x.shape


class TestDuplexAndTurnTaking:
    def test_duplex_barge_in(self):
        duplex = DuplexDialogueManager(barge_in_sensitivity=0.5)
        duplex.start_speaking()
        assert duplex.state == DialogueState.SPEAKING

        # User starts speaking loudly
        res = duplex.on_user_speech_frame(user_speaking_prob=0.8)
        assert res["interrupted"] is True
        assert duplex.state == DialogueState.INTERRUPTED

    def test_voice_activity_detector(self):
        vad = VoiceActivityDetector(energy_threshold=0.01)
        silence = torch.zeros(100)
        assert vad.is_speech(silence)["is_speech"] is False

        loud = torch.ones(100) * 0.5
        assert vad.is_speech(loud)["is_speech"] is True

    def test_end_of_utterance(self):
        eou = EndOfUtterancePredictor(silence_threshold_ms=400)
        assert eou.update(is_speech=False, chunk_duration_ms=200) is False
        assert eou.update(is_speech=False, chunk_duration_ms=200) is True  # Reached 400ms


class TestOmniGenerator:
    def test_generator_heads(self):
        cfg = OmniConfig(d_model=32, text_vocab_size=100, audio_codebook_size=64)
        gen = OmniGenerator(cfg)
        h = torch.randn(2, 5, 32)
        out = gen(h)
        assert out["text_logits"].shape == (2, 5, 100)
        assert out["speech_logits"].shape == (2, 5, 64)
        assert out["tool_logits"].shape == (2, 5, 2)


class TestNanoMindOmniModel:
    def test_model_forward_combined(self):
        cfg = OmniConfig(d_model=32, n_layers=2, n_heads=2, text_vocab_size=100, audio_codebook_size=64)
        model = NanoMindOmni(cfg)

        text = torch.randint(0, 100, (1, 3))
        img = torch.randn(1, 2, 256)
        audio = torch.randn(1, 4, 80)

        out = model(text_tokens=text, image_patches=img, audio_frames=audio)
        assert out["hidden_states"].shape == (1, 9, 32)
        assert out["text_logits"].shape == (1, 9, 100)
        assert out["speech_logits"].shape == (1, 9, 64)

    def test_omni_pipeline_chat(self):
        cfg = OmniConfig(d_model=32, n_layers=1, n_heads=2, text_vocab_size=50, audio_codebook_size=32)
        model = NanoMindOmni(cfg)
        pipeline = OmniPipeline(model)
        text = torch.randint(0, 50, (1, 2))
        resp = pipeline.chat(text_tokens=text)
        assert len(resp.text_token_ids) == 1
        assert len(resp.speech_token_ids) == 1
