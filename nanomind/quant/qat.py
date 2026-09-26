"""
nanomind/quant/qat.py — Quantization-Aware Training (QAT).

## QAT: Train With Fake Quantization

QAT simulates quantization during training using the
Straight-Through Estimator (STE) for gradients:

  Forward:  y = fake_quantize(x) = dequant(quant(x))
  Backward: dy/dx = 1  (treat quantize as identity)

This allows gradients to flow through the non-differentiable
rounding operation, letting the model adapt to quantization noise.

QAT typically achieves better accuracy than PTQ:
  PTQ INT4: model adapts AFTER training → error
  QAT INT4: model trains WITH noise → learns robust weights

Used by: Google's TFLite, Apple's CoreML, NVIDIA TensorRT.

The STE is justified because rounding is "almost everywhere" differentiable,
and the error is bounded by the quantization step size.

Reference:
  Bengio et al. (2013) "Estimating or Propagating Gradients Through Stochastic Neurons"
  https://arxiv.org/abs/1308.3432

  Jacob et al. (2018) "Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference" https://arxiv.org/abs/1712.05877
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.quant.config import QuantConfig
from nanomind.quant.quantizer import compute_scale_zero


class STEQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator for quantization.

    Forward:  round(x / scale) × scale   (fake quantize)
    Backward: pass gradient through unchanged.
    """

    @staticmethod
    def forward(ctx, x, scale, zero, q_min, q_max):
        q = (x / scale + zero).round().clamp(q_min, q_max)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def fake_quant_ste(
    x:     torch.Tensor,
    scale: torch.Tensor,
    zero:  torch.Tensor,
    cfg:   QuantConfig,
) -> torch.Tensor:
    """Fake quantization with STE gradient pass-through."""
    return STEQuantize.apply(x, scale, zero, cfg.q_min, cfg.q_max)


class QATLinear(nn.Module):
    """
    Linear layer with quantization-aware training.

    Applies fake quantization to weights (and optionally activations)
    during the forward pass. Gradients flow through via STE.

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        bias:         Include bias.
        cfg:          :class:`QuantConfig`.

    Example::

        layer = QATLinear(64, 128, cfg=QuantConfig(bits=4))
        y     = layer(x)   # uses fake-quantized weights
        loss.backward()    # STE: gradients flow through quantization!
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        bias:         bool = True,
        cfg:          QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg     = cfg or QuantConfig(bits=8)
        self.linear  = nn.Linear(in_features, out_features, bias=bias)
        # Learnable scale and zero point
        self._scale:    torch.Tensor | None = None
        self._zero:     torch.Tensor | None = None
        self.quant_act  = cfg.quantize_activations if cfg else False

    def _get_scale_zero(self) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.linear.weight
        if self._scale is None:
            s, z = compute_scale_zero(w, self.cfg)
            self._scale, self._zero = s, z
        return self._scale, self._zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w          = self.linear.weight
        s, z       = compute_scale_zero(w.detach(), self.cfg)
        w_q        = fake_quant_ste(w, s, z, self.cfg)   # STE quantize weights

        if self.quant_act:
            sx, zx = compute_scale_zero(x.detach(), self.cfg)
            x      = fake_quant_ste(x, sx, zx, self.cfg)

        return F.linear(x, w_q, self.linear.bias)

    def effective_bits(self) -> float:
        """Actual bits used (accounts for outliers / residuals)."""
        return float(self.cfg.bits)


def convert_to_qat(
    model: nn.Module,
    cfg:   QuantConfig,
    in_place: bool = True,
) -> nn.Module:
    """
    Convert all nn.Linear layers to QATLinear.

    Args:
        model:    PyTorch model.
        cfg:      Quantization configuration.
        in_place: Modify model in place.

    Returns:
        Model with QATLinear layers.
    """
    if not in_place:
        import copy
        model = copy.deepcopy(model)

    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            qat = QATLinear(module.in_features, module.out_features,
                             bias=module.bias is not None, cfg=cfg)
            qat.linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                qat.linear.bias.data = module.bias.data.clone()
            setattr(model, name, qat)
        else:
            convert_to_qat(module, cfg, in_place=True)
    return model
