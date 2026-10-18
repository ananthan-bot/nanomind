"""
nanomind/sae/trainer.py — SAE training loop, dead neuron tracking, and feature resampling.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from nanomind.sae.config import SAEConfig, SAEOutput


class DeadNeuronTracker:
    """Tracks activation frequencies across batches to detect dead features."""
    def __init__(self, d_sae: int, window: int = 100):
        self.d_sae = d_sae
        self.window = window
        self.activation_counts = torch.zeros(d_sae)
        self.total_tokens = 0

    def update(self, latents: torch.Tensor) -> None:
        """Update firing counts from batch latents (..., d_sae)."""
        active = (latents.reshape(-1, self.d_sae) > 0).float()
        self.activation_counts += active.sum(dim=0).cpu()
        self.total_tokens += active.shape[0]

    @property
    def dead_feature_indices(self) -> List[int]:
        """Return indices of features that never fired."""
        return (self.activation_counts == 0).nonzero().squeeze(-1).tolist()

    @property
    def dead_ratio(self) -> float:
        return float((self.activation_counts == 0).float().mean().item())


class SAETrainer:
    """End-to-end trainer managing gradient steps and decoder norm re-projection."""
    def __init__(self, sae: nn.Module, config: SAEConfig):
        self.sae = sae
        self.config = config
        self.optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr)
        self.tracker = DeadNeuronTracker(config.d_sae)

    def train_step(self, x: torch.Tensor) -> SAEOutput:
        self.optimizer.zero_grad()
        out = self.sae(x)
        out.loss.backward()
        self.optimizer.step()

        if hasattr(self.sae, "normalize_decoder_weights"):
            self.sae.normalize_decoder_weights()

        self.tracker.update(out.latents)
        return out
