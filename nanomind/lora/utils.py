"""
nanomind/lora/utils.py — LoRA parameter counting and inspection utilities.
"""

from __future__ import annotations

import torch.nn as nn

from nanomind.lora.layer import LoRALinear


def lora_parameter_stats(model: nn.Module) -> dict:
    """
    Count total, trainable, and LoRA-specific parameters.

    Args:
        model: A model with LoRA layers injected.

    Returns:
        Dict with:
        - ``total``     : total parameter count
        - ``trainable`` : trainable (LoRA) parameter count
        - ``frozen``    : frozen (base) parameter count
        - ``lora_pct``  : percentage of parameters that are trainable
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    lora_pct  = 100.0 * trainable / max(total, 1)
    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    frozen,
        "lora_pct":  lora_pct,
    }


def get_lora_state_dict(model: nn.Module) -> dict:
    """
    Extract only the LoRA-specific parameters from a model's state dict.

    Used for saving lightweight LoRA checkpoints (much smaller than full model).

    Args:
        model: A model with LoRA layers.

    Returns:
        State dict containing only ``lora_A`` and ``lora_B`` entries.
    """
    return {
        k: v
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def merge_all_lora(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA deltas into base weights across the entire model.

    After merging, the model runs as fast as the original with no overhead.

    Args:
        model: Model with LoRA layers.

    Returns:
        Same model with all LoRA weights merged (in-place).
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    return model


def unmerge_all_lora(model: nn.Module) -> nn.Module:
    """
    Unmerge all LoRA deltas from base weights across the entire model.

    Args:
        model: Model with merged LoRA layers.

    Returns:
        Same model with LoRA weights separated (in-place).
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
    return model


def print_lora_summary(model: nn.Module) -> None:
    """Print a summary of LoRA configuration and trainable parameters."""
    stats = lora_parameter_stats(model)
    print("=" * 50)
    print("LoRA Parameter Summary")
    print("=" * 50)
    print(f"  Total parameters  : {stats['total']:>12,}")
    print(f"  Trainable (LoRA)  : {stats['trainable']:>12,}  ({stats['lora_pct']:.2f}%)")
    print(f"  Frozen (base)     : {stats['frozen']:>12,}")
    print("=" * 50)

    # List LoRA layers
    lora_layers = [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, LoRALinear)
    ]
    print(f"  LoRA layers ({len(lora_layers)}):")
    for name, m in lora_layers:
        n = 2 * m.r * (m.in_features + m.out_features)
        print(f"    {name:<40} r={m.r}  params={n:,}")
    print("=" * 50)
