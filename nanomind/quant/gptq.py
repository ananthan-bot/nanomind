"""
nanomind/quant/gptq.py — GPTQ: Optimal Post-Training Quantization.

## GPTQ Algorithm (Frantar et al., 2022)

GPTQ quantizes each weight matrix W column by column,
compensating quantization error using the inverse Hessian.

Key insight: the second-order Taylor expansion of the loss gives:
  ΔL ≈ (1/2) δW^T H δW

where H = X^T X is the input Hessian (Gram matrix of activations).

Algorithm per layer:
  For each column (weight) j:
    1. q_j = quantize(w_j)             [quantize this weight]
    2. e_j = w_j - q_j                [quantization error]
    3. Update remaining weights:
       w_{j:} -= e_j × H^{-1}_{j,j:} / H^{-1}_{j,j}
    
This is equivalent to OBS (Optimal Brain Surgeon) applied column-wise.

Benefits:
  ✓ 4-bit quantization with minimal accuracy loss
  ✓ Supports per-group quantization (G=128)
  ✓ Works for very large models (65B+)

Used by: AutoGPTQ, ExLlama, llama.cpp (Q4_K_M format)

Reference:
  Frantar et al. (2022) https://arxiv.org/abs/2210.17323
"""

from __future__ import annotations
import torch
import torch.nn as nn
from nanomind.quant.config import QuantConfig
from nanomind.quant.quantizer import TensorQuantizer, compute_scale_zero, quantize, dequantize
from nanomind.utils.logger import get_logger

log = get_logger("quant.gptq")


class GPTQQuantizer:
    """
    GPTQ column-wise weight quantization with Hessian compensation.

    Args:
        W:    ``(d_out, d_in)`` weight matrix.
        H:    ``(d_in, d_in)`` Hessian (X^T X), from calibration data.
        cfg:  :class:`QuantConfig`.

    Example::

        gptq = GPTQQuantizer(W, H, QuantConfig(bits=4))
        W_q  = gptq.quantize()   # (d_out, d_in) quantized-dequantized weights
        err  = gptq.error
    """

    def __init__(
        self,
        W:   torch.Tensor,
        H:   torch.Tensor,
        cfg: QuantConfig,
        block_size: int = 128,
    ) -> None:
        self.W          = W.clone().float()
        self.H          = H.clone().float()
        self.cfg        = cfg
        self.block_size = block_size
        self.error:     float = 0.0

    def _invert_hessian(self) -> torch.Tensor:
        """Compute dampened inverse Hessian."""
        H    = self.H.clone()
        d    = H.shape[0]
        # Damping for numerical stability
        damp = 0.01 * H.diag().mean()
        H   += damp * torch.eye(d)
        try:
            H_inv = torch.linalg.inv(H)
        except RuntimeError:
            H_inv = torch.eye(d)
        return H_inv

    def quantize(self) -> torch.Tensor:
        """
        Run GPTQ column-wise quantization.

        Returns:
            ``(d_out, d_in)`` dequantized weight matrix with minimised error.
        """
        W     = self.W.clone()
        d_out, d_in = W.shape
        H_inv = self._invert_hessian()
        W_q   = torch.zeros_like(W)

        for col in range(d_in):
            w_col  = W[:, col]    # (d_out,)
            # Quantize this column
            s, z   = compute_scale_zero(w_col.unsqueeze(-1), self.cfg, dim=1)
            q_col  = quantize(w_col.unsqueeze(-1), s, z, self.cfg).squeeze(-1)
            wq_col = dequantize(q_col.unsqueeze(-1), s, z).squeeze(-1)
            W_q[:, col] = wq_col

            # Propagate error to remaining columns
            err    = w_col - wq_col   # (d_out,)
            if col + 1 < d_in and H_inv[col, col].abs() > 1e-10:
                update = err.unsqueeze(-1) @ H_inv[col, col+1:].unsqueeze(0) / H_inv[col, col]
                W[:, col+1:] -= update

        self.error = (W_q - self.W).pow(2).mean().item()
        return W_q

    def quantize_blocks(self) -> torch.Tensor:
        """Block-wise quantization for efficiency (block_size columns at a time)."""
        # Simplified: call column-wise quantize (full GPTQ uses Cholesky blocks)
        return self.quantize()


class HessianCollector:
    """
    Collect Hessian (X^T X) from forward passes for GPTQ.

    Args:
        module: nn.Linear to profile.

    Example::

        collector = HessianCollector(linear_layer)
        collector.enable()
        # Run calibration data through model...
        collector.disable()
        H = collector.hessian()
    """

    def __init__(self, module: nn.Linear) -> None:
        self.module = module
        self._H:    torch.Tensor | None = None
        self._n:    int = 0
        self._hook: object = None

    def enable(self) -> None:
        def _hook(mod, inp, out):
            x = inp[0].detach().view(-1, inp[0].shape[-1]).float()   # (B*T, D)
            H = x.T @ x
            self._H = H if self._H is None else self._H + H
            self._n += x.shape[0]
        self._hook = self.module.register_forward_hook(_hook)

    def disable(self) -> None:
        if self._hook:
            self._hook.remove()
            self._hook = None

    def hessian(self) -> torch.Tensor:
        """Return normalised Hessian: (X^T X) / n_samples."""
        if self._H is None:
            d = self.module.weight.shape[1]
            return torch.eye(d)
        return self._H / max(self._n, 1)

    def reset(self) -> None:
        self._H = None
        self._n = 0
