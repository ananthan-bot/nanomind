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
