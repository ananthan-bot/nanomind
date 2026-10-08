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
