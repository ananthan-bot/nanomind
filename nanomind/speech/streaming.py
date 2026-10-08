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
