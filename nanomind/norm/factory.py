"""nanomind/norm/factory.py — Normalization layer factory."""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def get_norm(name: str, d_model: int) -> nn.Module:
    """Return a normalization layer by name."""
    if name == "rmsnorm":
        return RMSNorm(d_model)
    return nn.LayerNorm(d_model)
