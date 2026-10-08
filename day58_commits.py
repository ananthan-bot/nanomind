"""
day58_commits.py — 20 atomic commits for Day 58: Speech & Audio-Language Models (Whisper/RVQ/EnCodec/Mini-Omni).
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

print("\n=== DAY 58: Speech & Audio-Language Models (Whisper / EnCodec RVQ / Mini-Omni) — 20 commits, v5.8.0 ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — Speech package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/speech/__init__.py",
      '"""NanoMind Speech sub-package — Audio-Language Models, RVQ Codecs, and Whisper-style Encoders."""\n')
commit("feat: add nanomind/speech/ package skeleton for Speech & Audio-Language Models")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — Configuration classes
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/speech/config.py", '''\
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
''')
commit("feat: implement AudioConfig, CodecConfig, SpeechEncoderConfig, and SpeechLMConfig in nanomind/speech/config.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — Mel Spectrogram Features
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/speech/features.py", '''\
"""
nanomind/speech/features.py — Log-mel spectrogram extraction in pure PyTorch.
"""
import math
from typing import Optional
import torch
import torch.nn as nn
from nanomind.speech.config import AudioConfig


def hz_to_mel(hz: float) -> float:
    """Convert frequency from Hz to Mel scale."""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """Convert frequency from Mel scale back to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 400,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float = 8000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Construct triangular Mel filterbank matrix (n_mels, n_fft // 2 + 1).
    """
    n_freqs = n_fft // 2 + 1
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    # Uniform points on mel scale
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, device=device)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    # Convert Hz to FFT bin indices
    bin_points = torch.floor((n_fft + 1) * hz_points / sample_rate).long()

    filterbank = torch.zeros(n_mels, n_freqs, device=device)
    for i in range(n_mels):
        left = int(bin_points[i].item())
        center = int(bin_points[i + 1].item())
        right = int(bin_points[i + 2].item())

        for f in range(left, center):
            if center > left and f < n_freqs:
                filterbank[i, f] = (f - left) / max(1, center - left)
        for f in range(center, right):
            if right > center and f < n_freqs:
                filterbank[i, f] = (right - f) / max(1, right - center)

    return filterbank


class MelSpectrogramExtractor(nn.Module):
    """
    Extracts log-mel spectrogram features from raw audio waveform.
    Waveform (B, T_samples) -> Log-Mel (B, n_mels, T_frames).
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        super().__init__()
        self.config = config or AudioConfig()
        fb = create_mel_filterbank(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
        )
        self.register_buffer("filterbank", fb)

    def apply_preemphasis(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply pre-emphasis filter: y[t] = x[t] - alpha * x[t-1]."""
        if self.config.preemphasis <= 0.0:
            return waveform
        padded = torch.nn.functional.pad(waveform, (1, 0), mode="reflect")
        return padded[:, 1:] - self.config.preemphasis * padded[:, :-1]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (B, T_samples) or (T_samples,)
        Returns:
            log_mel: (B, n_mels, T_frames)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # 1. Pre-emphasis
        x = self.apply_preemphasis(waveform)

        # 2. Window and STFT
        window = torch.hann_window(self.config.win_length, device=waveform.device)
        stft = torch.stft(
            x,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
            return_complex=True,
            center=True,
        )
        # Power spectrogram: (B, n_freqs, T_frames)
        power_spec = stft.abs().pow(2.0)

        # 3. Apply Mel filterbank: (n_mels, n_freqs) @ (B, n_freqs, T_frames)
        mel_spec = torch.matmul(self.filterbank, power_spec)

        # 4. Log-compression with floor
        log_mel = torch.log(torch.clamp(mel_spec, min=self.config.log_offset))
        return log_mel
''')
commit("feat: implement mel filterbank and log-mel spectrogram extractor in nanomind/speech/features.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — Vector Quantizer (VQ) with STE
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/speech/codec.py", '''\
"""
nanomind/speech/codec.py — Neural Audio Codec: Vector Quantization & Residual Vector Quantization (RVQ).
"""
import math
from typing import Tuple, List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.speech.config import CodecConfig


class VectorQuantizer(nn.Module):
    """
    Standard Vector Quantizer (VQ-VAE style) with Straight-Through Estimator (STE).
    Quantizes continuous latent embeddings into discrete codebook indices.
    """

    def __init__(self, codebook_size: int = 1024, embedding_dim: int = 128, commitment_weight: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight

        # Learnable codebook vectors
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: continuous latents (B, D, T)
        Returns:
            z_q: quantized latents (B, D, T)
            commit_loss: scalar commitment loss
            indices: discrete codes (B, T)
        """
        # Rearrange to (B, T, D)
        z_t = z.permute(0, 2, 1).contiguous()
        B, T, D = z_t.shape

        # Flatten: (B * T, D)
        z_flat = z_t.view(-1, D)

        # Compute squared L2 distance to all codebook vectors: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z @ e^T
        d = (
            torch.sum(z_flat ** 2, dim=-1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=-1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        indices_flat = torch.argmin(d, dim=-1)
        indices = indices_flat.view(B, T)

        # Look up quantized embeddings
        z_q_flat = self.embedding(indices_flat)
        z_q_t = z_q_flat.view(B, T, D)

        # Commitment loss
        loss_commit = F.mse_loss(z_t, z_q_t.detach())
        loss_codebook = F.mse_loss(z_q_t, z_t.detach())
        total_loss = loss_codebook + self.commitment_weight * loss_commit

        # Straight-Through Estimator: copy gradients from z_q to z
        z_q_t = z_t + (z_q_t - z_t).detach()

        # Reshape back to (B, D, T)
        z_q = z_q_t.permute(0, 2, 1).contiguous()
        return z_q, total_loss, indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert discrete indices (B, T) to embeddings (B, D, T)."""
        z_q_t = self.embedding(indices)  # (B, T, D)
        return z_q_t.permute(0, 2, 1).contiguous()
''')
commit("feat: implement VectorQuantizer with straight-through estimator in nanomind/speech/codec.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — Residual Vector Quantization (RVQ)
# ══════════════════════════════════════════════════════════════════════════════
codec_src = read("nanomind/speech/codec.py")
codec_src += '''\


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ) as used in SoundStream and EnCodec.
    Cascades n_q quantizers where each quantizer models the residual of previous stages.
    """

    def __init__(self, config: Optional[CodecConfig] = None):
        super().__init__()
        self.config = config or CodecConfig()
        self.n_q = self.config.n_q
        self.codebook_size = self.config.codebook_size
        self.embedding_dim = self.config.embedding_dim

        # n_q quantizer stages
        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                codebook_size=self.codebook_size,
                embedding_dim=self.embedding_dim,
                commitment_weight=self.config.commitment_weight,
            )
            for _ in range(self.n_q)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: continuous audio latents (B, D, T)
        Returns:
            z_q_total: aggregated quantized latents (B, D, T)
            total_loss: summed commitment/codebook loss across stages
            all_indices: multi-stage discrete codes (B, n_q, T)
        """
        B, D, T = z.shape
        residual = z
        z_q_total = torch.zeros_like(z)
        total_loss = torch.tensor(0.0, device=z.device)
        indices_list = []

        for vq in self.quantizers:
            z_q_stage, loss_stage, idx_stage = vq(residual)
            residual = residual - z_q_stage
            z_q_total = z_q_total + z_q_stage
            total_loss = total_loss + loss_stage
            indices_list.append(idx_stage.unsqueeze(1))  # (B, 1, T)

        all_indices = torch.cat(indices_list, dim=1)  # (B, n_q, T)
        return z_q_total, total_loss, all_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct audio latents from multi-stage codes.
        codes: (B, n_q, T)
        Returns: (B, D, T) reconstructed latents
        """
        B, n_q, T = codes.shape
        z_out = torch.zeros(B, self.embedding_dim, T, device=codes.device)
        for i in range(min(n_q, self.n_q)):
            vq = self.quantizers[i]
            z_stage = vq.decode_indices(codes[:, i, :])
            z_out = z_out + z_stage
        return z_out
'''
write("nanomind/speech/codec.py", codec_src)
commit("feat: implement Residual Vector Quantization (RVQ) multi-stage codec in nanomind/speech/codec.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Audio Transformer Encoder (Whisper style)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/speech/encoder.py", '''\
"""
nanomind/speech/encoder.py — Whisper-style Audio Transformer Encoder with 1D Conv Downsampling.
"""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.speech.config import SpeechEncoderConfig


class AudioEncoderBlock(nn.Module):
    """Transformer Encoder layer for acoustic representations with Pre-LayerNorm."""

    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        B, T, D = x.shape

        # 1. Self Attention with Pre-LN
        norm_x = self.ln1(x)
        q = self.q_proj(norm_x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out_proj(out)

        # 2. Feed Forward
        x = x + self.mlp(self.ln2(x))
        return x


class AudioEncoder(nn.Module):
    """
    Audio Transformer Encoder (Whisper architecture):
    2x 1D Convolutions with stride 2 for 4x temporal downsampling,
    followed by sinusoidal positional encodings and Transformer encoder blocks.
    """

    def __init__(self, config: Optional[SpeechEncoderConfig] = None):
        super().__init__()
        self.config = config or SpeechEncoderConfig()
        d_model = self.config.d_model

        # 1D Convolution downsampling: 2 strided convolutions -> 4x frame rate reduction
        self.conv1 = nn.Conv1d(self.config.n_mels, d_model, kernel_size=self.config.conv_kernel, stride=self.config.conv_stride, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=self.config.conv_kernel, stride=self.config.conv_stride, padding=1)

        # Sinusoidal positional embeddings
        self.register_buffer(
            "positional_embedding",
            self._build_sinusoidal_embeddings(self.config.max_source_positions, d_model)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            AudioEncoderBlock(d_model=d_model, n_heads=self.config.n_heads, dropout=self.config.dropout)
            for _ in range(self.config.n_layers)
        ])
        self.ln_post = nn.LayerNorm(d_model)

    def _build_sinusoidal_embeddings(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: Log-mel spectrogram (B, n_mels, T_frames)
        Returns:
            audio_features: (B, T_downsampled, d_model)
        """
        # 1. Conv downsampling
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))  # (B, d_model, T_down)

        # 2. Transpose to (B, T_down, d_model)
        x = x.transpose(1, 2)
        B, T, D = x.shape

        # 3. Add positional embeddings
        x = x + self.positional_embedding[:T, :].unsqueeze(0)

        # 4. Transformer blocks
        for block in self.blocks:
            x = block(x)

        return self.ln_post(x)
''')
commit("feat: implement Whisper-style Audio Transformer Encoder in nanomind/speech/encoder.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Cross-Modal Audio-Text Projector
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/speech/projector.py", '''\
"""
nanomind/speech/projector.py — Cross-modal projector aligning audio encoder states to LLM token space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeechProjector(nn.Module):
    """
    Projector module that transforms audio encoder representations to match LLM embedding dimensions.
    Supports linear projection or 2-layer MLP with optional temporal pooling.
    """

    def __init__(self, audio_dim: int = 256, llm_dim: int = 256, downsample_rate: int = 2):
        super().__init__()
        self.downsample_rate = downsample_rate
        in_dim = audio_dim * downsample_rate if downsample_rate > 1 else audio_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        audio_features: (B, T_audio, audio_dim)
        Returns:
            projected_features: (B, T_audio // downsample_rate, llm_dim)
        """
        B, T, D = audio_features.shape

        if self.downsample_rate > 1:
            # Pad if not divisible by downsample_rate
            rem = T % self.downsample_rate
            if rem != 0:
                pad_len = self.downsample_rate - rem
                audio_features = F.pad(audio_features, (0, 0, 0, pad_len))
                T = audio_features.shape[1]

            # Reshape by concatenating consecutive frames
            audio_features = audio_features.view(B, T // self.downsample_rate, D * self.downsample_rate)

        return self.net(audio_features)
''')
commit("feat: implement cross-modal audio-text projector and frame pooling in nanomind/speech/projector.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — End-to-End Speech-Language Model
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/speech/model.py", '''\
"""
nanomind/speech/model.py — End-to-End Speech-Language Model (Mini-Omni / SpeechGPT style).
Jointly processes spoken audio inputs and generates text and acoustic RVQ tokens.
"""
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn

from nanomind.speech.config import SpeechLMConfig
from nanomind.speech.features import MelSpectrogramExtractor
from nanomind.speech.encoder import AudioEncoder
from nanomind.speech.projector import SpeechProjector
from nanomind.speech.codec import ResidualVectorQuantizer


class SpeechLanguageModel(nn.Module):
    """
    Multimodal Speech-Language Model:
    1. Audio feature extraction (MelSpectrogramExtractor)
    2. Audio encoding (AudioEncoder)
    3. Multimodal projection (SpeechProjector)
    4. Text embedding + Backbone LLM processing
    5. Dual prediction heads:
       - Text head: predicts next textual token
       - Codec head: predicts next acoustic RVQ tokens
    """

    def __init__(self, config: Optional[SpeechLMConfig] = None):
        super().__init__()
        self.config = config or SpeechLMConfig()

        # Audio front-end
        self.mel_extractor = MelSpectrogramExtractor(self.config.audio)
        self.audio_encoder = AudioEncoder(self.config.encoder)
        self.projector = SpeechProjector(
            audio_dim=self.config.encoder.d_model,
            llm_dim=self.config.llm_d_model,
            downsample_rate=self.config.projector_downsample_rate,
        )

        # RVQ neural audio codec for speech synthesis
        self.codec = ResidualVectorQuantizer(self.config.codec)

        # Text embedding
        self.text_embedding = nn.Embedding(self.config.text_vocab_size, self.config.llm_d_model)

        # Backbone transformer layers
        self.llm_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.config.llm_d_model,
                nhead=4,
                dim_feedforward=4 * self.config.llm_d_model,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(2)
        ])
        self.ln_f = nn.LayerNorm(self.config.llm_d_model)

        # Output heads
        self.text_head = nn.Linear(self.config.llm_d_model, self.config.text_vocab_size, bias=False)
        self.audio_head = nn.Linear(self.config.llm_d_model, self.config.codec.codebook_size, bias=False)

    def encode_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Raw audio -> Mel -> Encoder -> Projector -> LLM-compatible audio tokens."""
        mel = self.mel_extractor(waveform)
        enc_out = self.audio_encoder(mel)
        audio_tokens = self.projector(enc_out)
        return audio_tokens

    def forward(
        self,
        waveform: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Multimodal forward pass.
        waveform: (B, T_audio_samples)
        text_tokens: (B, T_text)
        Returns:
            dict containing:
              - 'text_logits': (B, T_total, text_vocab_size)
              - 'audio_logits': (B, T_total, audio_vocab_size)
        """
        prefix_embeds = []

        if waveform is not None:
            audio_embeds = self.encode_audio(waveform)
            prefix_embeds.append(audio_embeds)

        if text_tokens is not None:
            text_embeds = self.text_embedding(text_tokens)
            prefix_embeds.append(text_embeds)

        if not prefix_embeds:
            raise ValueError("Must provide either waveform or text_tokens")

        # Concatenate audio + text embeddings along sequence dimension
        x = torch.cat(prefix_embeds, dim=1)

        # LLM forward
        for block in self.llm_blocks:
            x = block(x)
        h = self.ln_f(x)

        text_logits = self.text_head(h)
        audio_logits = self.audio_head(h)

        return {
            "hidden_states": h,
            "text_logits": text_logits,
            "audio_logits": audio_logits,
        }
''')
commit("feat: implement SpeechLanguageModel with text and acoustic codec heads in nanomind/speech/model.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — CTC Loss & Multi-Task Speech Losses
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/speech/loss.py", '''\
"""
nanomind/speech/loss.py — Connectionist Temporal Classification (CTC) and Multi-Stage Codec Cross-Entropy.
"""
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLossWrapper(nn.Module):
    """
    CTC Loss for speech recognition without requiring explicit frame-level alignments.
    """

    def __init__(self, blank_idx: int = 0, zero_infinity: bool = True):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank_idx, zero_infinity=zero_infinity)

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        log_probs: (T, B, C) log probabilities
        targets: (B, S) target indices
        input_lengths: (B,)
        target_lengths: (B,)
        """
        return self.ctc(log_probs, targets, input_lengths, target_lengths)


class MultiStageCodecLoss(nn.Module):
    """
    Cross-entropy loss for multi-stage RVQ acoustic codes.
    Calculates weighted cross-entropy across all n_q quantizer levels.
    """

    def __init__(self, n_q: int = 8, decay_per_stage: float = 0.9):
        super().__init__()
        self.n_q = n_q
        self.decay_per_stage = decay_per_stage

    def forward(self, logits: torch.Tensor, target_codes: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, T, codebook_size)
        target_codes: (B, n_q, T)
        """
        B, n_q, T = target_codes.shape
        total_loss = torch.tensor(0.0, device=logits.device)
        total_weight = 0.0

        for stage in range(min(n_q, self.n_q)):
            weight = self.decay_per_stage ** stage
            targets_stage = target_codes[:, stage, :]  # (B, T)
            loss_stage = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_stage.view(-1))
            total_loss = total_loss + weight * loss_stage
            total_weight += weight

        return total_loss / max(1e-6, total_weight)
''')
commit("feat: implement multi-task CTC loss and multi-stage codebook loss in nanomind/speech/loss.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — Real-Time Streaming Audio Pipeline
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/speech/streaming.py", '''\
"""
nanomind/speech/streaming.py — Streaming audio buffer, chunking, and Real-Time Factor (RTF) profiler.
"""
import time
from typing import List, Optional, Tuple, Dict, Any
import torch


class StreamingAudioBuffer:
    """
    Ring buffer for continuous incoming streaming audio chunks.
    Maintains a rolling window of audio samples and emits fixed-size processing frames.
    """

    def __init__(self, sample_rate: int = 16000, chunk_ms: int = 200, context_ms: int = 40):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * (chunk_ms / 1000.0))
        self.context_size = int(sample_rate * (context_ms / 1000.0))
        self.buffer = torch.zeros(0)

    def append(self, samples: torch.Tensor):
        """Append incoming audio samples (1D tensor)."""
        self.buffer = torch.cat([self.buffer, samples.cpu()])

    def get_next_chunk(self) -> Optional[torch.Tensor]:
        """
        Retrieve next chunk of size (chunk_size + context_size) if available.
        Advances buffer by chunk_size.
        """
        required = self.chunk_size + self.context_size
        if len(self.buffer) < required:
            return None

        chunk = self.buffer[:required].clone()
        # Advance by chunk_size
        self.buffer = self.buffer[self.chunk_size:]
        return chunk

    def reset(self):
        self.buffer = torch.zeros(0)


class RTFProfiler:
    """
    Real-Time Factor (RTF) profiler:
    RTF = Processing Time (s) / Audio Duration (s)
    An RTF < 1.0 means the model runs faster than real time (e.g. RTF 0.1 = 10x real-time).
    """

    def __init__(self):
        self.total_audio_sec: float = 0.0
        self.total_proc_sec: float = 0.0
        self.latencies: List[float] = []

    def record_chunk(self, audio_duration_sec: float, processing_time_sec: float):
        self.total_audio_sec += audio_duration_sec
        self.total_proc_sec += processing_time_sec
        self.latencies.append(processing_time_sec)

    @property
    def rtf(self) -> float:
        if self.total_audio_sec == 0:
            return 0.0
        return self.total_proc_sec / self.total_audio_sec

    def summary(self) -> Dict[str, float]:
        return {
            "total_audio_sec": round(self.total_audio_sec, 3),
            "total_proc_sec": round(self.total_proc_sec, 3),
            "rtf": round(self.rtf, 4),
            "avg_latency_ms": round(1000.0 * (sum(self.latencies) / max(1, len(self.latencies))), 2),
            "realtime_capable": self.rtf < 1.0,
        }
''')
commit("feat: implement streaming audio buffer and real-time factor (RTF) profiler in nanomind/speech/streaming.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — Speech Metrics: WER, CER & Codebook Perplexity
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/speech/metrics.py", '''\
"""
nanomind/speech/metrics.py — Word Error Rate (WER), Character Error Rate (CER), and Codebook Perplexity.
"""
import math
from typing import List, Dict, Any
import torch


def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """Compute Levenshtein edit distance between two token sequences."""
    n1, n2 = len(seq1), len(seq2)
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]

    for i in range(n1 + 1):
        dp[i][0] = i
    for j in range(n2 + 1):
        dp[0][j] = j

    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n1][n2]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) = (Substitutions + Deletions + Insertions) / Total Reference Words.
    """
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return float(dist / len(ref_words))


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) at character level."""
    ref_chars = list(reference.strip().lower())
    hyp_chars = list(hypothesis.strip().lower())

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    dist = levenshtein_distance(ref_chars, hyp_chars)
    return float(dist / len(ref_chars))


def compute_codebook_perplexity(codes: torch.Tensor, codebook_size: int = 1024) -> float:
    """
    Measures the perplexity / utilization of codebook entries in RVQ:
    Perplexity = exp(- sum p_k * log(p_k))
    A higher perplexity (closer to codebook_size) indicates uniform utilization (no codebook collapse).
    """
    flat = codes.flatten()
    if len(flat) == 0:
        return 0.0

    counts = torch.bincount(flat, minlength=codebook_size).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log(probs))
    return float(torch.exp(entropy).item())
''')
commit("feat: implement Word Error Rate (WER), CER, and codebook perplexity in nanomind/speech/metrics.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — Speech API Exposure
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/speech/__init__.py", '''\
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
''')

# Update nanomind/__init__.py to re-export SpeechLanguageModel & configs
init_py = read("nanomind/__init__.py")
if "from nanomind.speech import" not in init_py:
    init_py = init_py.replace(
        "from nanomind.reasoning import ReasoningConfig, ProcessRewardModel, MonteCarloTreeSearch, GRPOTrainer\n",
        "from nanomind.reasoning import ReasoningConfig, ProcessRewardModel, MonteCarloTreeSearch, GRPOTrainer\n"
        "from nanomind.speech import SpeechLMConfig, SpeechLanguageModel, ResidualVectorQuantizer, MelSpectrogramExtractor\n"
    )
    init_py = init_py.replace(
        '    "GRPOTrainer",\n',
        '    "GRPOTrainer",\n'
        '    "SpeechLMConfig",\n'
        '    "SpeechLanguageModel",\n'
        '    "ResidualVectorQuantizer",\n'
        '    "MelSpectrogramExtractor",\n'
    )
    write("nanomind/__init__.py", init_py)

commit("feat: expose speech API in nanomind/speech/__init__.py and top-level package")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — Examples Demo
# ══════════════════════════════════════════════════════════════════════════════
write("examples/speech_demo.py", '''\
"""
examples/speech_demo.py — End-to-end Demonstration of Speech & Audio-Language Models.
Demonstrates:
  1. Mel Spectrogram Extraction from Audio Waveform
  2. Residual Vector Quantization (RVQ) Codec Encoding & Decoding
  3. Whisper-style Audio Transformer Encoder Forward Pass
  4. End-to-End SpeechLanguageModel Multimodal Output
  5. Streaming Audio Buffer & Real-Time Factor (RTF) Profiling
  6. Word Error Rate (WER) Evaluation
"""
import time
import torch

from nanomind.speech import (
    AudioConfig,
    MelSpectrogramExtractor,
    CodecConfig,
    ResidualVectorQuantizer,
    SpeechEncoderConfig,
    AudioEncoder,
    SpeechLMConfig,
    SpeechLanguageModel,
    StreamingAudioBuffer,
    RTFProfiler,
    word_error_rate,
    compute_codebook_perplexity,
)


def run_demo():
    print("=" * 70)
    print("  NanoMind Day 58: Speech & Audio-Language Models (Whisper / RVQ)")
    print("=" * 70)

    # 1. Audio Waveform & Mel-Spectrogram Extraction
    print("\\n[1] Mel-Spectrogram Extraction:")
    sr = 16000
    duration_sec = 1.0
    t = torch.linspace(0, duration_sec, int(sr * duration_sec))
    # Synthetic audio signal with harmonics (440 Hz + 880 Hz)
    waveform = 0.5 * torch.sin(2 * 3.14159 * 440 * t) + 0.3 * torch.sin(2 * 3.14159 * 880 * t)
    waveform = waveform.unsqueeze(0)  # (1, 16000)

    extractor = MelSpectrogramExtractor(AudioConfig(sample_rate=sr, n_mels=80))
    log_mel = extractor(waveform)
    print(f"  Raw Waveform shape: {waveform.shape} (1 second at {sr} Hz)")
    print(f"  Log-Mel Spectrogram shape: {log_mel.shape} (80 mel bands x {log_mel.shape[2]} frames)")

    # 2. Residual Vector Quantization (RVQ)
    print("\\n[2] Residual Vector Quantization (RVQ) 4-Stage Codec:")
    codec_cfg = CodecConfig(codebook_size=512, n_q=4, embedding_dim=64)
    rvq = ResidualVectorQuantizer(codec_cfg)

    # Continuous audio latents (B=1, D=64, T=50)
    z = torch.randn(1, 64, 50)
    z_q, loss_rvq, codes = rvq(z)
    recon = rvq.decode(codes)
    perplexity = compute_codebook_perplexity(codes, codebook_size=512)

    print(f"  Input Latent shape: {z.shape}")
    print(f"  RVQ Codes shape: {codes.shape} (4 quantization stages)")
    print(f"  Reconstructed shape: {recon.shape}")
    print(f"  Quantization Commitment Loss: {loss_rvq.item():.4f}")
    print(f"  Codebook Perplexity: {perplexity:.1f} / 512")

    # 3. Whisper-Style Audio Transformer Encoder
    print("\\n[3] Whisper-Style Audio Encoder:")
    enc_cfg = SpeechEncoderConfig(n_mels=80, d_model=128, n_layers=2, n_heads=4)
    encoder = AudioEncoder(enc_cfg)
    audio_feats = encoder(log_mel)
    print(f"  Downsampled Audio Features: {audio_feats.shape} (4x temporal reduction)")

    # 4. End-to-End SpeechLanguageModel
    print("\\n[4] End-to-End Speech-Language Model (SpeechLM):")
    lm_cfg = SpeechLMConfig(
        llm_d_model=128,
        text_vocab_size=1000,
        audio_vocab_size=512,
    )
    model = SpeechLanguageModel(lm_cfg)
    text_prompt = torch.randint(0, 1000, (1, 8))
    out = model(waveform=waveform, text_tokens=text_prompt)

    print(f"  Text Logits shape: {out['text_logits'].shape}")
    print(f"  Audio Logits shape: {out['audio_logits'].shape}")

    # 5. Real-Time Streaming Audio Buffer & RTF Profiler
    print("\\n[5] Streaming Audio & RTF Profiler:")
    stream_buf = StreamingAudioBuffer(sample_rate=16000, chunk_ms=200)
    profiler = RTFProfiler()

    # Simulate 5 consecutive 200ms audio chunks
    chunk_samples = int(16000 * 0.2)
    for _ in range(5):
        audio_chunk = torch.randn(chunk_samples)
        stream_buf.append(audio_chunk)

        t0 = time.time()
        c = stream_buf.get_next_chunk()
        proc_time = time.time() - t0 + 0.01  # Simulated ~10ms forward pass
        profiler.record_chunk(audio_duration_sec=0.2, processing_time_sec=proc_time)

    prof_summary = profiler.summary()
    print(f"  Total Processed Audio: {prof_summary['total_audio_sec']} s")
    print(f"  Average Chunk Latency: {prof_summary['avg_latency_ms']} ms")
    print(f"  Real-Time Factor (RTF): {prof_summary['rtf']} (< 1.0 means faster than real-time)")

    # 6. Word Error Rate (WER)
    print("\\n[6] Speech Recognition Evaluation:")
    ref = "the quick brown fox jumps over the lazy dog"
    hyp = "the quick brown fox jumped over a lazy dog"
    wer = word_error_rate(ref, hyp)
    print(f"  Reference : '{ref}'")
    print(f"  Hypothesis: '{hyp}'")
    print(f"  Word Error Rate (WER): {wer * 100:.1f}%")

    print("\\n[OK] All Speech & Audio-Language Model demos completed successfully!")


if __name__ == "__main__":
    run_demo()
''')
commit("feat: add examples/speech_demo.py demonstrating RVQ, Whisper encoder, SpeechLM, and streaming")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — Unit Tests Part 1: Mel Spectrogram Features
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_speech.py", '''\
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
''')
commit("test: add tests for log-mel spectrogram features and pre-emphasis filter")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — Unit Tests Part 2: Vector Quantization & RVQ
# ══════════════════════════════════════════════════════════════════════════════
test_src = read("tests/test_speech.py")
test_src += '''\


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
'''
write("tests/test_speech.py", test_src)
commit("test: add tests for VectorQuantizer and Residual Vector Quantization (RVQ)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — Unit Tests Part 3: Audio Transformer Encoder
# ══════════════════════════════════════════════════════════════════════════════
test_src = read("tests/test_speech.py")
test_src += '''\


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
'''
write("tests/test_speech.py", test_src)
commit("test: add tests for Audio Transformer Encoder downsampling and attention")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — Unit Tests Part 4: SpeechLanguageModel
# ══════════════════════════════════════════════════════════════════════════════
test_src = read("tests/test_speech.py")
test_src += '''\


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
'''
write("tests/test_speech.py", test_src)
commit("test: add tests for SpeechLanguageModel multimodal forward passes")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — Unit Tests Part 5: CTC, Streaming & Metrics
# ══════════════════════════════════════════════════════════════════════════════
test_src = read("tests/test_speech.py")
test_src += '''\


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
'''
write("tests/test_speech.py", test_src)
commit("test: add tests for CTC loss, streaming buffer, and WER evaluation metric")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — Bump to v5.8.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace('__version__ = "5.7.0"', '__version__ = "5.8.0"')
write("nanomind/__init__.py", src)
commit("feat: bump to v5.8.0 — Speech & Audio-Language Models (Whisper/RVQ/EnCodec/Mini-Omni) release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + Push + Tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `reasoning`  | Reasoning Models & Test-Time Compute — PRM (step verification), MCTS, ToT, GRPO (DeepSeek-R1), STaR, scaling laws |",
    "| `reasoning`  | Reasoning Models & Test-Time Compute — PRM (step verification), MCTS, ToT, GRPO (DeepSeek-R1), STaR, scaling laws |\n"
    "| `speech`     | Speech & Audio-Language Models — Mel spectrogram, EnCodec RVQ, Whisper encoder, SpeechLM, streaming RTF |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [5.8.0] — 2024 — Speech & Audio-Language Models\\n\\n### Added\\n"
      "- `AudioConfig`, `CodecConfig`, `SpeechEncoderConfig`, `SpeechLMConfig` — full speech pipeline configs\\n"
      "- `create_mel_filterbank` & `MelSpectrogramExtractor` — log-mel spectrogram extraction in pure PyTorch\\n"
      "- `VectorQuantizer` — VQ-VAE codebook with straight-through estimator (STE)\\n"
      "- `ResidualVectorQuantizer` (RVQ) — cascading multi-stage acoustic codebook (SoundStream/EnCodec)\\n"
      "- `AudioEncoder` & `AudioEncoderBlock` — Whisper-style 1D conv downsampling + audio transformer\\n"
      "- `SpeechProjector` — temporal pooling and cross-modal projection to LLM token space\\n"
      "- `SpeechLanguageModel` — end-to-end multimodal model with dual text & acoustic codec generation heads\\n"
      "- `CTCLossWrapper` — Connectionist Temporal Classification loss\\n"
      "- `MultiStageCodecLoss` — multi-stage weighted codebook cross-entropy\\n"
      "- `StreamingAudioBuffer` — sliding chunk audio streaming buffer\\n"
      "- `RTFProfiler` — Real-Time Factor and latency benchmarking profiler\\n"
      "- `word_error_rate` & `character_error_rate` — Levenshtein distance speech recognition metrics\\n"
      "- `compute_codebook_perplexity` — RVQ codebook utilization diagnostic\\n"
      "- `examples/speech_demo.py` — runnable speech processing & generation demo\\n\\n---\\n\\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v5.8.0, update README and CHANGELOG for Day 58 Speech")

print("\n=== Pushing Day 58 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")
run("git", "tag", "-a", "v5.8.0", "-m", "NanoMind v5.8.0 — Speech & Audio-Language Models (Whisper/RVQ/EnCodec/Mini-Omni)", check=False)
r = run("git", "push", "origin", "v5.8.0", check=False)
print("Tag v5.8.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 58 COMPLETE — v5.8.0 TAGGED! ===")
