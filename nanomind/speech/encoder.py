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
