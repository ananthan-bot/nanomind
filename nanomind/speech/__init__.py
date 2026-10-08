"""
nanomind.speech — Speech & Audio-Language Models (Whisper / EnCodec RVQ / Mini-Omni).
"""
from nanomind.speech.config import (
    AudioConfig,
    CodecConfig,
    SpeechEncoderConfig,
    SpeechLMConfig,
)
from nanomind.speech.features import (
    hz_to_mel,
    mel_to_hz,
    create_mel_filterbank,
    MelSpectrogramExtractor,
)
from nanomind.speech.codec import (
    VectorQuantizer,
    ResidualVectorQuantizer,
)
from nanomind.speech.encoder import (
    AudioEncoderBlock,
    AudioEncoder,
)
from nanomind.speech.projector import (
    SpeechProjector,
)
from nanomind.speech.model import (
    SpeechLanguageModel,
)
from nanomind.speech.loss import (
    CTCLossWrapper,
    MultiStageCodecLoss,
)
from nanomind.speech.streaming import (
    StreamingAudioBuffer,
    RTFProfiler,
)
from nanomind.speech.metrics import (
    word_error_rate,
    character_error_rate,
    compute_codebook_perplexity,
)

__all__ = [
    "AudioConfig",
    "CodecConfig",
    "SpeechEncoderConfig",
    "SpeechLMConfig",
    "hz_to_mel",
    "mel_to_hz",
    "create_mel_filterbank",
    "MelSpectrogramExtractor",
    "VectorQuantizer",
    "ResidualVectorQuantizer",
    "AudioEncoderBlock",
    "AudioEncoder",
    "SpeechProjector",
    "SpeechLanguageModel",
    "CTCLossWrapper",
    "MultiStageCodecLoss",
    "StreamingAudioBuffer",
    "RTFProfiler",
    "word_error_rate",
    "character_error_rate",
    "compute_codebook_perplexity",
]
