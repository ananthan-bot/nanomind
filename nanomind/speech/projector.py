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
