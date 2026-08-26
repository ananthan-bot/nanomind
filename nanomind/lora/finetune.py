"""
nanomind/lora/finetune.py — Convenience helpers for LoRA fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanomind.lora.config import LoRAConfig
from nanomind.lora.model import LoRAModel
from nanomind.optim import get_optimizer, get_lr_scheduler
from nanomind.utils.logger import get_logger

log = get_logger("lora.finetune")


def finetune_with_lora(
    base_model: nn.Module,
    train_loader: DataLoader,
    lora_cfg: LoRAConfig | None = None,
    lr: float = 3e-4,
    max_iters: int = 1000,
    device: torch.device | None = None,
    save_path: str | None = None,
) -> LoRAModel:
    """
    Fine-tune a pre-trained model with LoRA.

    A convenience function that:
    1. Wraps the model with :class:`LoRAModel`
    2. Creates an AdamW optimizer on LoRA parameters only
    3. Runs a simple training loop
    4. Optionally saves LoRA weights

    Args:
        base_model:   Pre-trained model to fine-tune.
        train_loader: DataLoader yielding (x, y) batches.
        lora_cfg:     LoRA configuration (defaults to r=8, alpha=16).
        lr:           Learning rate for LoRA parameters.
        max_iters:    Number of fine-tuning steps.
        device:       Training device.
        save_path:    If provided, save LoRA weights here after training.

    Returns:
        Trained :class:`LoRAModel`.
    """
    lora_cfg = lora_cfg or LoRAConfig()
    device   = device or next(base_model.parameters()).device

    lora_model = LoRAModel(base_model, lora_cfg)
    lora_model.to(device)
    lora_model.summary()

    optimizer = torch.optim.AdamW(lora_model.lora_parameters(), lr=lr)
    loader_iter = iter(train_loader)
    step = 0

    while step < max_iters:
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss = lora_model(x, y)
        loss.backward()
        optimizer.step()
        step += 1

        if step % 100 == 0 or step == max_iters:
            log.info(f"step {step}/{max_iters}  loss={loss.item():.4f}")

    if save_path:
        lora_model.save(save_path, metadata={"steps": max_iters, "lr": lr})

    return lora_model
