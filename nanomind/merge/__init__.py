"""
nanomind.merge — Model Merging & Weight Averaging (TIES / DARE / SLERP / Soups).
"""
from nanomind.merge.config import (
    MergeMethod,
    MergeConfig,
)
from nanomind.merge.task_vectors import (
    TaskVector,
)
from nanomind.merge.ties import (
    TIESMerger,
)
from nanomind.merge.dare import (
    DAREPruner,
)
from nanomind.merge.slerp import (
    slerp,
)
from nanomind.merge.frankenmerge import (
    LayerSplicer,
)
from nanomind.merge.soup import (
    ModelSoup,
)
from nanomind.merge.merger import (
    ModelMerger,
)
from nanomind.merge.metrics import (
    compute_parameter_drift,
    compute_task_vector_cosine_similarity,
)

__all__ = [
    "MergeMethod",
    "MergeConfig",
    "TaskVector",
    "TIESMerger",
    "DAREPruner",
    "slerp",
    "LayerSplicer",
    "ModelSoup",
    "ModelMerger",
    "compute_parameter_drift",
    "compute_task_vector_cosine_similarity",
]
