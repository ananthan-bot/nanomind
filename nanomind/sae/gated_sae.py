"""
nanomind/sae/gated_sae.py — Gated Sparse Autoencoder.
Decouples feature detection (gating path) from feature magnitude (magnitude path).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.sae.config import SAEConfig, SAEOutput


class GatedSAE(nn.Module):
    """
    Gated SAE:
    pi = Heaviside(W_gate x + b_gate)
    m = ReLU(W_mag x + b_mag)
    f = pi * m
    """
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.b_dec = nn.Parameter(torch.zeros(config.d_in))
        # Gating sub-network
        self.w_gate = nn.Linear(config.d_in, config.d_sae, bias=True)
        # Magnitude sub-network
        self.w_mag = nn.Linear(config.d_in, config.d_sae, bias=True)
        # Reconstruction decoder
        self.w_dec = nn.Linear(config.d_sae, config.d_in, bias=False)

        if config.normalize_decoder:
            self.normalize_decoder_weights()

    def normalize_decoder_weights(self) -> None:
        with torch.no_grad():
            self.w_dec.weight.data = F.normalize(self.w_dec.weight.data, p=2, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.b_dec
        gate = (self.w_gate(centered) > 0).float()
        mag = F.relu(self.w_mag(centered))
        return gate * mag

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.w_dec(latents) + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        latents = self.encode(x)
        reconstruction = self.decode(latents)

        mse = F.mse_loss(reconstruction, x)
        l1 = latents.sum(dim=-1).mean()
        loss = mse + self.config.l1_coeff * l1
        l0 = (latents > 0).float().sum(dim=-1).mean().item()

        return SAEOutput(
            reconstruction=reconstruction,
            latents=latents,
            loss=loss,
            l0=l0,
            l1=l1.item(),
            mse=mse.item(),
        )
