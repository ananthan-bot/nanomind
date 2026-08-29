"""
nanomind/quant/quantize.py — Model-level quantization: replace Linear with QuantizedLinear.
"""

from __future__ import annotations

import torch.nn as nn

from nanomind.quant.config import QuantConfig
from nanomind.quant.layer import QuantizedLinear
from nanomind.quant.dynamic import DynamicQuantizedLinear
from nanomind.utils.logger import get_logger

log = get_logger("quant.quantize")


def quantize_model(
    model: nn.Module,
    cfg:   QuantConfig | None = None,
) -> nn.Module:
    """
    Replace ``nn.Linear`` layers with quantized equivalents in-place.

    Iterates over all modules and replaces any ``nn.Linear`` whose parent
    attribute name is not in ``cfg.skip_modules`` with a ``QuantizedLinear``
    (or ``DynamicQuantizedLinear`` if ``cfg.mode == "dynamic"``).

    Args:
        model: The model to quantize (modified in-place).
        cfg:   Quantization configuration.

    Returns:
        The quantized model (same reference, modified in-place).

    Example::

        cfg = QuantConfig(mode="weight_only", granularity="per_channel")
        quantize_model(model, cfg)
        # model's Linear layers are now QuantizedLinear
    """
    cfg = cfg or QuantConfig()
    layer_cls = (
        DynamicQuantizedLinear
        if cfg.mode == "dynamic"
        else QuantizedLinear
    )

    n_quantized = 0
    for parent_name, module in model.named_modules():
        for attr_name in list(vars(module).keys()):
            child = getattr(module, attr_name, None)
            if not isinstance(child, nn.Linear):
                continue
            # Skip modules matching any skip pattern
            full_name = f"{parent_name}.{attr_name}" if parent_name else attr_name
            if any(skip in full_name for skip in cfg.skip_modules):
                log.debug(f"  Skipping {full_name} (in skip_modules)")
                continue

            ql = layer_cls.from_linear(child, granularity=cfg.granularity)
            setattr(module, attr_name, ql)
            n_quantized += 1
            log.debug(f"  Quantized: {full_name}")

    log.info(
        f"Quantized {n_quantized} Linear layers "
        f"(mode={cfg.mode}, granularity={cfg.granularity})"
    )
    return model
