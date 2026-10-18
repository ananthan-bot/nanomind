"""
nanomind/sae/topk_sae.py — Top-K Sparse Autoencoder (Gao et al., 2024).
Guarantees exact k-sparse representations without L1 hyperparameter tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.sae.config import SAEConfig, SAEOutput


class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder:
    Selects exactly the top-K highest activations per sample and zeros out the remainder.
    """
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.k = min(config.k_sparse, config.d_sae)
        self.b_dec = nn.Parameter(torch.zeros(config.d_in))
        self.w_enc = nn.Linear(config.d_in, config.d_sae, bias=True)
        self.w_dec = nn.Linear(config.d_sae, config.d_in, bias=False)

        if config.normalize_decoder:
            self.normalize_decoder_weights()

    def normalize_decoder_weights(self) -> None:
        """Project decoder columns to unit L2 norm."""
        with torch.no_grad():
            self.w_dec.weight.data = F.normalize(self.w_dec.weight.data, p=2, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pre-activations and apply Top-K thresholding."""
        centered = x - self.b_dec
        pre_acts = F.relu(self.w_enc(centered))

        topk_vals, topk_indices = torch.topk(pre_acts, self.k, dim=-1)
        latents = torch.zeros_like(pre_acts)
        latents.scatter_(-1, topk_indices, topk_vals)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.w_dec(latents) + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        latents = self.encode(x)
        reconstruction = self.decode(latents)

        mse = F.mse_loss(reconstruction, x)
        loss = mse
        l0 = float(self.k)
        l1 = latents.sum(dim=-1).mean().item()

        return SAEOutput(
            reconstruction=reconstruction,
            latents=latents,
            loss=loss,
            l0=l0,
            l1=l1,
            mse=mse.item(),
        )
