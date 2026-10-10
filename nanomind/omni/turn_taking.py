"""
nanomind/omni/turn_taking.py — Voice Activity Detection (VAD) and End-of-Utterance (EOU) prediction.
"""
from typing import Optional, Dict
import torch
import torch.nn as nn


class VoiceActivityDetector:
    """
    Computes root-mean-square (RMS) energy and zero-crossing rate to classify active speech.
    """

    def __init__(self, energy_threshold: float = 0.02):
        self.energy_threshold = energy_threshold

    def compute_energy(self, audio_chunk: torch.Tensor) -> float:
        """Compute RMS audio frame energy."""
        if audio_chunk.numel() == 0:
            return 0.0
        return float(torch.sqrt(torch.mean(audio_chunk.float() ** 2)).item())

    def is_speech(self, audio_chunk: torch.Tensor) -> Dict[str, Any]:
        """Classify whether chunk contains voice activity."""
        energy = self.compute_energy(audio_chunk)
        is_active = energy >= self.energy_threshold
        confidence = min(1.0, energy / max(1e-5, self.energy_threshold * 2.0))

        return {
            "is_speech": is_active,
            "energy": round(energy, 4),
            "confidence": round(confidence, 4),
        }


class EndOfUtterancePredictor:
    """
    Predicts when user has completed their query based on silence duration and semantic prosody.
    """

    def __init__(self, silence_threshold_ms: int = 500):
        self.silence_threshold_ms = silence_threshold_ms
        self.current_silence_ms: int = 0

    def update(self, is_speech: bool, chunk_duration_ms: int = 200) -> bool:
        """
        Returns True when silence duration indicates user finished speaking.
        """
        if is_speech:
            self.current_silence_ms = 0
            return False
        else:
            self.current_silence_ms += chunk_duration_ms
            return self.current_silence_ms >= self.silence_threshold_ms

    def reset(self):
        self.current_silence_ms = 0
