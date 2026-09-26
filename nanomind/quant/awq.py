"""
nanomind/quant/awq.py — AWQ: Activation-Aware Weight Quantization.

## AWQ (Lin et al., 2023)

Key insight: not all weights are equally important!
  Weights corresponding to large activation channels matter more.

Instead of correcting for quantization error (GPTQ),
AWQ scales important weight channels UP before quantization:
  w̃_c = w_c × s_c   (scale important channels)
  x̃_c = x_c / s_c   (compensate by scaling activations down)

The scale s_c is chosen to minimise quantization error for important channels.
Optimal scale: s* = mean(|x_c|)^α / mean(|w_c|)^{1-α}

With α=0.5: geometric mean of activation and weight magnitudes.
This is applied BEFORE standard quantization (RTN or GPTQ).

Benefits:
  ✓ No matrix inversion needed (faster than GPTQ)
  ✓ Works well with grouped quantization (G=128)
  ✓ Simple implementation — just scale + quantize

AWQ achieves similar quality to GPTQ with less computation.

Reference:
  Lin et al. (2023) "AWQ: Activation-Aware Weight Quantization for LLM Compression"
  https://arxiv.org/abs/2306.00978
"""

from __future__ import annotations
import torch
import torch.nn as nn
from nanomind.quant.config import QuantConfig
from nanomind.quant.quantizer import TensorQuantizer
from nanomind.utils.logger import get_logger

log = get_logger("quant.awq")


class AWQQuantizer:
    """
    Activation-Aware Weight Quantization.

    Scales weight channels by activation magnitudes before quantization.

    Args:
        W:           ``(d_out, d_in)`` weight matrix.
        act_scales:  ``(d_in,)`` mean absolute activation magnitudes per channel.
        cfg:         :class:`QuantConfig`.
        alpha:       Scaling exponent (0=weight-only, 1=activation-only, 0.5=balanced).

    Example::

        awq      = AWQQuantizer(W, act_scales, QuantConfig(bits=4))
        W_q, s   = awq.quantize()   # quantized weights and scaling factors
    """

    def __init__(
        self,
        W:          torch.Tensor,
        act_scales: torch.Tensor,
        cfg:        QuantConfig,
        alpha:      float = 0.5,
    ) -> None:
        self.W          = W.clone().float()
        self.act_scales = act_scales.float()
        self.cfg        = cfg
        self.alpha      = alpha

    def _compute_scale(self) -> torch.Tensor:
        """
        Compute per-channel AWQ scale.

        s_c = act_scales^α / w_mean^{1-α}
        """
        # Mean absolute weight per input channel (d_in,)
        w_scale = self.W.abs().mean(dim=0)
        # AWQ scale
        s = (self.act_scales ** self.alpha) / (w_scale ** (1 - self.alpha) + 1e-8)
        # Normalise so scale doesn't inflate weights overall
        s = s / s.mean()
        return s.clamp(min=1e-4)

    def quantize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply AWQ scaling and quantize.

        Returns:
            ``(W_quantized, scale_factors)``
            W_quantized is the dequantized weight for drop-in use.
        """
        s      = self._compute_scale()        # (d_in,)
        W_s    = self.W * s.unsqueeze(0)      # scale important channels

        q      = TensorQuantizer(self.cfg)
        q.calibrate(W_s)
        W_q    = q.fake_quantize(W_s)         # QDQ in scaled space

        # Undo scaling: divide by s to restore correct magnitude
        W_out  = W_q / s.unsqueeze(0)
        return W_out, s

    def error(self) -> dict:
        """Measure AWQ quantization error."""
        W_q, s = self.quantize()
        diff   = W_q - self.W
        return {
            "mse":    diff.pow(2).mean().item(),
            "max":    diff.abs().max().item(),
            "scale_mean": s.mean().item(),
            "scale_std":  s.std().item(),
        }


class ActivationScaleCollector:
    """
    Collect mean absolute activation magnitudes per channel.

    Used to compute AWQ scales.

    Args:
        module: nn.Linear to profile.
    """

    def __init__(self, module: nn.Linear) -> None:
        self.module  = module
        self._scales: torch.Tensor | None = None
        self._n:      int = 0
        self._hook:   object = None

    def enable(self) -> None:
        def _hook(mod, inp, out):
            x = inp[0].detach().view(-1, inp[0].shape[-1])
            s = x.abs().mean(dim=0)
            self._scales = s if self._scales is None else self._scales + s
            self._n     += 1
        self._hook = self.module.register_forward_hook(_hook)

    def disable(self) -> None:
        if self._hook:
            self._hook.remove()
            self._hook = None

    def scales(self) -> torch.Tensor:
        """Return mean absolute activation magnitudes: ``(d_in,)``."""
        if self._scales is None:
            return torch.ones(self.module.weight.shape[1])
        return self._scales / max(self._n, 1)

    def reset(self) -> None:
        self._scales = None
        self._n      = 0
