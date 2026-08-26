"""
nanomind/lora/model.py — High-level LoRA model wrapper.

LoRAModel wraps a pre-trained NanoMind model and handles the full
LoRA lifecycle:
  1. Inject LoRA layers into target modules
  2. Freeze base parameters
  3. Train only LoRA parameters
  4. Optionally merge for inference
  5. Save/load lightweight LoRA checkpoints
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.lora.config import LoRAConfig
from nanomind.lora.inject import inject_lora, mark_only_lora_as_trainable
from nanomind.lora.checkpoint import save_lora_checkpoint, load_lora_checkpoint
from nanomind.lora.utils import (
    lora_parameter_stats,
    merge_all_lora,
    unmerge_all_lora,
    print_lora_summary,
)
from nanomind.utils.logger import get_logger


class LoRAModel(nn.Module):
    """
    LoRA-wrapped NanoMind model for parameter-efficient fine-tuning.

    Usage::

        base_model = NanoMind(cfg)
        lora_cfg   = LoRAConfig(r=8, alpha=16, target_modules=["q_proj","v_proj"])
        model      = LoRAModel(base_model, lora_cfg)

        # Only LoRA parameters are updated
        optimizer = torch.optim.AdamW(model.lora_parameters(), lr=3e-4)

        # After fine-tuning, save only LoRA weights (~few MB)
        model.save("my_lora.pt")

    Args:
        model: Pre-trained base model.
        cfg:   LoRA configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: LoRAConfig,
    ) -> None:
        super().__init__()
        self.cfg  = cfg
        self.log  = get_logger("lora.model")

        # Inject LoRA layers
        inject_lora(model, cfg)
        mark_only_lora_as_trainable(model, bias=cfg.bias)
        self.model = model

        stats = lora_parameter_stats(model)
        self.log.info(
            f"LoRAModel ready: {stats['trainable']:,} trainable params "
            f"({stats['lora_pct']:.2f}% of {stats['total']:,} total)"
        )

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)

    def lora_parameters(self):
        """Return only the trainable LoRA parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def merge_for_inference(self) -> "LoRAModel":
        """Merge LoRA weights into base weights for zero-overhead inference."""
        merge_all_lora(self.model)
        self.log.info("LoRA weights merged — ready for inference.")
        return self

    def unmerge(self) -> "LoRAModel":
        """Unmerge LoRA weights for further training."""
        unmerge_all_lora(self.model)
        return self

    def save(self, path: str, metadata: dict | None = None) -> None:
        """Save only LoRA weights (lightweight checkpoint)."""
        save_lora_checkpoint(self.model, path, metadata)

    def load(self, path: str, device: torch.device | None = None) -> dict:
        """Load LoRA weights from a checkpoint."""
        return load_lora_checkpoint(self.model, path, device)

    def summary(self) -> None:
        """Print a LoRA parameter summary."""
        print_lora_summary(self.model)

    def __repr__(self) -> str:
        stats = lora_parameter_stats(self.model)
        return (
            f"LoRAModel("
            f"r={self.cfg.r}, "
            f"alpha={self.cfg.alpha}, "
            f"targets={self.cfg.target_modules}, "
            f"trainable={stats['trainable']:,} ({stats['lora_pct']:.2f}%))"
        )
