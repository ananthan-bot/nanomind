"""
nanomind/speech/config.py — Configuration dataclasses for audio feature extraction, codecs, and speech LMs.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AudioConfig:
    """Configuration for raw audio processing and spectrogram generation."""
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    preemphasis: float = 0.97
    log_offset: float = 1e-5


@dataclass
class CodecConfig:
    """Configuration for Residual Vector Quantization (RVQ) neural audio codec."""
    codebook_size: int = 1024
    n_q: int = 8               # Number of cascading residual quantizers
    embedding_dim: int = 128   # Latent dimension per quantized vector
    commitment_weight: float = 0.25
    decay: float = 0.99        # EMA decay for codebook vectors


@dataclass
class SpeechEncoderConfig:
    """Configuration for Whisper-style Transformer Audio Encoder."""
    n_mels: int = 80
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    conv_kernel: int = 3
    conv_stride: int = 2
    dropout: float = 0.1
    max_source_positions: int = 1500


@dataclass
class SpeechLMConfig:
    """Configuration for end-to-end Speech-Language Model (SpeechGPT / Mini-Omni style)."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)
    encoder: SpeechEncoderConfig = field(default_factory=SpeechEncoderConfig)
    llm_d_model: int = 256
    text_vocab_size: int = 50257
    audio_vocab_size: int = 1024
    projector_downsample_rate: int = 2
    streaming_chunk_ms: int = 200
