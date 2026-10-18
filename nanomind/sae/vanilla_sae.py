"""
nanomind/sae/vanilla_sae.py — Standard ReLU Sparse Autoencoder with L1 sparsity penalty.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.sae.config import SAEConfig, SAEOutput


class VanillaSAE(nn.Module):
    """
    Standard Sparse Autoencoder:
    latents = ReLU(W_enc (x - b_dec) + b_enc)
    reconstruction = W_dec latents + b_dec
    """
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.b_dec = nn.Parameter(torch.zeros(config.d_in))
        self.w_enc = nn.Linear(config.d_in, config.d_sae, bias=True)
        self.w_dec = nn.Linear(config.d_sae, config.d_in, bias=False)

        if config.normalize_decoder:
            self.normalize_decoder_weights()

    def normalize_decoder_weights(self) -> None:
        """Constrain decoder columns (features) to unit L2 norm."""
        with torch.no_grad():
            self.w_dec.weight.data = F.normalize(self.w_dec.weight.data, p=2, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations into sparse dictionary latents."""
        centered = x - self.b_dec
        return F.relu(self.w_enc(centered))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct input activations from sparse latents."""
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
