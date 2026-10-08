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
