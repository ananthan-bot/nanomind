"""
nanomind/sae/jumprelu_sae.py — JumpReLU Sparse Autoencoder (Rajamanoharan et al., DeepMind 2024).
Uses discontinuous Heaviside step thresholding to eliminate activation shrinkage.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.sae.config import SAEConfig, SAEOutput


class JumpReLUSAE(nn.Module):
    """
    JumpReLU SAE:
    f(z, theta) = z * (z > theta)
    Permits large feature magnitudes without shrinkage bias.
    """
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.b_dec = nn.Parameter(torch.zeros(config.d_in))
        self.w_enc = nn.Linear(config.d_in, config.d_sae, bias=True)
        self.w_dec = nn.Linear(config.d_sae, config.d_in, bias=False)
        # Learnable or fixed positive thresholds
        self.threshold = nn.Parameter(torch.full((config.d_sae,), config.threshold))

        if config.normalize_decoder:
            self.normalize_decoder_weights()

    def normalize_decoder_weights(self) -> None:
        with torch.no_grad():
            self.w_dec.weight.data = F.normalize(self.w_dec.weight.data, p=2, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.b_dec
        pre_acts = self.w_enc(centered)
        # JumpReLU: step thresholding
        mask = (pre_acts > self.threshold).float()
        return pre_acts * mask

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.w_dec(latents) + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        latents = self.encode(x)
        reconstruction = self.decode(latents)

        mse = F.mse_loss(reconstruction, x)
        l0_val = (latents > 0).float().sum(dim=-1).mean()
        loss = mse + self.config.l1_coeff * l0_val

        return SAEOutput(
            reconstruction=reconstruction,
            latents=latents,
            loss=loss,
            l0=l0_val.item(),
            l1=latents.sum(dim=-1).mean().item(),
            mse=mse.item(),
        )
