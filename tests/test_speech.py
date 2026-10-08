"""
tests/test_speech.py — Unit tests for Speech & Audio-Language Models.
"""
import math
import torch
from nanomind.speech import (
    AudioConfig,
    hz_to_mel,
    mel_to_hz,
    create_mel_filterbank,
    MelSpectrogramExtractor,
    CodecConfig,
    VectorQuantizer,
    ResidualVectorQuantizer,
    SpeechEncoderConfig,
    AudioEncoder,
    SpeechProjector,
    SpeechLMConfig,
    SpeechLanguageModel,
    CTCLossWrapper,
    MultiStageCodecLoss,
    StreamingAudioBuffer,
    RTFProfiler,
    word_error_rate,
    character_error_rate,
    compute_codebook_perplexity,
)


class TestAudioFeatures:
    def test_hz_mel_roundtrip(self):
        hz = 1000.0
        mel = hz_to_mel(hz)
        hz_recovered = mel_to_hz(mel)
        assert math.isclose(hz, hz_recovered, rel_tol=1e-4)

    def test_filterbank_shape(self):
        fb = create_mel_filterbank(sample_rate=16000, n_fft=400, n_mels=80)
        assert fb.shape == (80, 201)
        assert (fb >= 0.0).all()

    def test_mel_spectrogram_extractor(self):
        extractor = MelSpectrogramExtractor(AudioConfig(sample_rate=16000, n_mels=80))
        waveform = torch.randn(1, 16000)
        mel = extractor(waveform)
        assert mel.dim() == 3
        assert mel.shape[0] == 1
        assert mel.shape[1] == 80
        assert mel.shape[2] > 0


class TestVectorQuantization:
    def test_single_vector_quantizer(self):
        vq = VectorQuantizer(codebook_size=64, embedding_dim=16)
        z = torch.randn(2, 16, 20)
        z_q, loss, indices = vq(z)
        assert z_q.shape == z.shape
        assert indices.shape == (2, 20)
        assert loss.item() >= 0.0

    def test_residual_vector_quantizer(self):
        cfg = CodecConfig(codebook_size=128, n_q=4, embedding_dim=32)
        rvq = ResidualVectorQuantizer(cfg)
        z = torch.randn(2, 32, 15)
        z_q, loss, codes = rvq(z)
        assert z_q.shape == z.shape
        assert codes.shape == (2, 4, 15)

        recon = rvq.decode(codes)
        assert recon.shape == z.shape

    def test_codebook_perplexity(self):
        codes = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        perp = compute_codebook_perplexity(codes, codebook_size=4)
        assert math.isclose(perp, 4.0, rel_tol=1e-3)


class TestAudioEncoder:
    def test_audio_encoder_downsampling(self):
        cfg = SpeechEncoderConfig(n_mels=80, d_model=64, n_layers=2, n_heads=2)
        encoder = AudioEncoder(cfg)
        # 100 frames input
        mel = torch.randn(2, 80, 100)
        out = encoder(mel)
        # 2x stride-2 convs -> 25 frames
        assert out.shape == (2, 25, 64)

    def test_speech_projector(self):
        proj = SpeechProjector(audio_dim=64, llm_dim=128, downsample_rate=2)
        x = torch.randn(2, 20, 64)
        out = proj(x)
        assert out.shape == (2, 10, 128)


class TestSpeechLanguageModel:
    def test_multimodal_forward(self):
        cfg = SpeechLMConfig(
            llm_d_model=64,
            text_vocab_size=256,
            audio_vocab_size=128,
        )
        model = SpeechLanguageModel(cfg)
        waveform = torch.randn(1, 8000)  # 0.5 sec
        text_tokens = torch.randint(0, 256, (1, 6))

        out = model(waveform=waveform, text_tokens=text_tokens)
        assert "text_logits" in out
        assert "audio_logits" in out
        assert out["text_logits"].shape[-1] == 256
        assert out["audio_logits"].shape[-1] == 128


class TestLossStreamingAndMetrics:
    def test_multi_stage_codec_loss(self):
        loss_fn = MultiStageCodecLoss(n_q=4)
        logits = torch.randn(2, 10, 64)
        targets = torch.randint(0, 64, (2, 4, 10))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0

    def test_streaming_audio_buffer(self):
        buf = StreamingAudioBuffer(sample_rate=16000, chunk_ms=200, context_ms=40)
        # 16000 * 0.24 = 3840 samples required
        samples = torch.randn(4000)
        buf.append(samples)
        chunk = buf.get_next_chunk()
        assert chunk is not None
        assert len(chunk) == 3840

    def test_word_error_rate(self):
        ref = "hello world"
        hyp = "hello world"
        assert word_error_rate(ref, hyp) == 0.0

        hyp_err = "hello there"
        assert word_error_rate(ref, hyp_err) == 0.5

    def test_character_error_rate(self):
        ref = "cat"
        hyp = "bat"
        assert math.isclose(character_error_rate(ref, hyp), 1.0 / 3.0, rel_tol=1e-4)

    def test_rtf_profiler(self):
        prof = RTFProfiler()
        prof.record_chunk(audio_duration_sec=1.0, processing_time_sec=0.1)
        assert math.isclose(prof.rtf, 0.1, rel_tol=1e-4)
        assert prof.summary()["realtime_capable"] is True
