"""NanoMind LoRA (Low-Rank Adaptation) fine-tuning sub-package.

LoRA enables efficient fine-tuning by injecting trainable low-rank matrices
into frozen pre-trained linear layers. Only ~1-5% of parameters are updated.

Primary exports:
    - :class:`LoRAModel`              — high-level LoRA wrapper (recommended entry point)
    - :class:`LoRAConfig`             — rank, alpha, target_modules configuration
    - :class:`LoRALinear`             — drop-in LoRA-augmented nn.Linear

Injection:
    - :func:`inject_lora`             — replace target layers with LoRALinear
    - :func:`mark_only_lora_as_trainable` — freeze all base parameters

Checkpointing:
    - :func:`save_lora_checkpoint`    — save only A/B matrices (lightweight)
    - :func:`load_lora_checkpoint`    — load LoRA weights into injected model

Utilities:
    - :func:`lora_parameter_stats`    — count total/trainable/frozen params
    - :func:`get_lora_state_dict`     — extract LoRA-only state dict
    - :func:`merge_all_lora`          — merge deltas into weights (inference)
    - :func:`unmerge_all_lora`        — separate LoRA from base weights
    - :func:`print_lora_summary`      — pretty-print parameter table

Fine-tuning:
    - :func:`finetune_with_lora`      — one-call fine-tuning helper
"""

from nanomind.lora.config import LoRAConfig
from nanomind.lora.layer import LoRALinear
from nanomind.lora.inject import inject_lora, mark_only_lora_as_trainable
from nanomind.lora.checkpoint import save_lora_checkpoint, load_lora_checkpoint
from nanomind.lora.utils import (
    lora_parameter_stats,
    get_lora_state_dict,
    merge_all_lora,
    unmerge_all_lora,
    print_lora_summary,
)
from nanomind.lora.model import LoRAModel
from nanomind.lora.finetune import finetune_with_lora

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "LoRAModel",
    "inject_lora",
    "mark_only_lora_as_trainable",
    "save_lora_checkpoint",
    "load_lora_checkpoint",
    "lora_parameter_stats",
    "get_lora_state_dict",
    "merge_all_lora",
    "unmerge_all_lora",
    "print_lora_summary",
    "finetune_with_lora",
]
