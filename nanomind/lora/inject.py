"""
nanomind/lora/inject.py — Inject LoRA layers into a pre-trained model.
"""

from __future__ import annotations

import torch.nn as nn

from nanomind.lora.config import LoRAConfig
from nanomind.lora.layer import LoRALinear
from nanomind.utils.logger import get_logger

log = get_logger("lora.inject")


def inject_lora(
    model: nn.Module,
    cfg: LoRAConfig,
) -> nn.Module:
    """
    Replace target ``nn.Linear`` layers with :class:`~nanomind.lora.LoRALinear`.

    Only layers whose **name** (the attribute name in the parent module)
    matches one of ``cfg.target_modules`` are replaced.

    Args:
        model: The pre-trained model to inject LoRA into.
        cfg:   LoRA configuration.

    Returns:
        The same model with LoRA layers injected (in-place modification).

    Example::

        cfg = LoRAConfig(r=8, alpha=16, target_modules=["q_proj", "v_proj"])
        inject_lora(model, cfg)
    """
    n_injected = 0
    for name, module in model.named_modules():
        for attr_name in list(vars(module).keys()):
            child = getattr(module, attr_name, None)
            if not isinstance(child, nn.Linear):
                continue
            if attr_name not in cfg.target_modules:
                continue

            lora_layer = LoRALinear.from_linear(
                child,
                r=cfg.r,
                alpha=cfg.alpha,
                dropout=cfg.dropout,
            )
            setattr(module, attr_name, lora_layer)
            n_injected += 1
            log.debug(f"  Injected LoRA into: {name}.{attr_name}")

    log.info(f"LoRA injected into {n_injected} layers.")
    return model


def mark_only_lora_as_trainable(
    model: nn.Module,
    bias: str = "none",
) -> nn.Module:
    """
    Freeze all model parameters except LoRA matrices (and optionally biases).

    Args:
        model: Model with LoRA layers already injected.
        bias:  ``"none"`` — no biases trained
               ``"all"``  — all biases trained
               ``"lora_only"`` — only biases of LoRA layers trained

    Returns:
        The model with frozen parameters (in-place).
    """
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
        elif bias == "all" and "bias" in name:
            param.requires_grad_(True)
        elif bias == "lora_only" and "lora" in name and "bias" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    return model
