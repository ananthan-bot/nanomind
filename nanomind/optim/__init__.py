"""NanoMind optimizer and LR scheduler sub-package.

Optimizers:
    - :func:`get_optimizer`   — build AdamW/SGD with proper weight decay grouping
    - :func:`get_param_groups`— split params into decay / no-decay

LR Schedules (all callable: ``lr = schedule(step)``):
    - :class:`ConstantLR`     — fixed LR
    - :class:`LinearWarmup`   — linear ramp to max_lr
    - :class:`CosineDecay`    — cosine annealing
    - :class:`WarmupCosine`   — warmup + cosine (default)
    - :class:`LinearDecay`    — linear annealing
    - :class:`WarmupLinear`   — warmup + linear decay
    - :func:`get_lr_scheduler`— build a schedule by name
    - :func:`list_schedules`  — list all registered schedule names

Utilities:
    - :func:`compute_grad_norm` — global gradient L2 norm
    - :func:`get_grad_stats`    — gradient min/max/mean/norm
    - :func:`optimizer_summary` — human-readable optimizer info
    - :func:`schedule_preview`  — preview LR values at N steps
"""

from nanomind.optim.optimizer import get_optimizer
from nanomind.optim.param_groups import get_param_groups, count_param_groups
from nanomind.optim.schedules import (
    LRSchedule,
    ConstantLR,
    LinearWarmup,
    CosineDecay,
    WarmupCosine,
    LinearDecay,
    WarmupLinear,
    get_lr_scheduler,
    list_schedules,
)
from nanomind.optim.grad_utils import compute_grad_norm, get_grad_stats
from nanomind.optim.summary import optimizer_summary, schedule_preview

__all__ = [
    "get_optimizer",
    "get_param_groups",
    "count_param_groups",
    "LRSchedule",
    "ConstantLR",
    "LinearWarmup",
    "CosineDecay",
    "WarmupCosine",
    "LinearDecay",
    "WarmupLinear",
    "get_lr_scheduler",
    "list_schedules",
    "compute_grad_norm",
    "get_grad_stats",
    "optimizer_summary",
    "schedule_preview",
]
