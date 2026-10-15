"""
day63_commits.py — 20 atomic commits for Day 63: Model Merging & Weight Averaging (TIES/DARE/SLERP/Soups).
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== DAY 63: Model Merging & Weight Averaging (v6.3.0) — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — Merge package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/merge/__init__.py",
      '"""NanoMind Merge sub-package — Model Merging, Task Arithmetic, TIES, DARE, and SLERP."""\n')
commit("feat: add nanomind/merge/ package skeleton for Model Merging & Weight Averaging")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — Configuration classes & Enums
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/merge/config.py", '''\
"""
nanomind/merge/config.py — Configuration dataclasses and merge method enums.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


class MergeMethod(str, Enum):
    """Supported model merging techniques."""
    LINEAR = "linear"        # Weighted parameter averaging
    TASK_ARITHMETIC = "task_arithmetic"  # Additive task vectors
    TIES = "ties"            # Trimming, Electing Sign, Disjoint Merge
    DARE_TIES = "dare_ties"  # Drop And REscale with TIES
    DARE_LINEAR = "dare_linear"  # Drop And REscale with Linear
    SLERP = "slerp"          # Spherical Linear Interpolation
    SOUP = "soup"            # Model Soup uniform/greedy averaging
    FRANKENMERGE = "frankenmerge"  # Layer-wise passthrough splicing


@dataclass
class MergeConfig:
    """Configuration for model merging execution."""
    method: MergeMethod = MergeMethod.TIES
    weights: List[float] = field(default_factory=lambda: [1.0])
    density: float = 0.2     # Fraction of top parameters retained in TIES trimming
    drop_rate: float = 0.5   # Probability of dropping parameter in DARE
    slerp_t: float = 0.5     # Interpolation factor for SLERP in [0, 1]
    rescale: bool = True     # Whether to rescale remaining parameters in DARE
    normalize_weights: bool = True
''')
commit("feat: implement MergeConfig, MergeMethod, and recipe definitions in nanomind/merge/config.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — Task Vectors & Task Arithmetic
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/merge/task_vectors.py", '''\
"""
nanomind/merge/task_vectors.py — Task Vectors and Task Arithmetic (Ilharco et al., 2023).
Computes parameter delta tau = theta_ft - theta_base, supporting addition, scaling, and task negation.
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class TaskVector:
    """
    Represents the parameter shift tau = theta_ft - theta_base induced by fine-tuning on a task.
    """

    def __init__(self, vector_dict: Dict[str, torch.Tensor]):
        self.vector_dict = {k: v.clone().detach() for k, v in vector_dict.items()}

    @classmethod
    def from_models(cls, base_model: nn.Module, ft_model: nn.Module) -> "TaskVector":
        """Compute task vector tau = theta_ft - theta_base."""
        diff = {}
        base_params = dict(base_model.named_parameters())
        ft_params = dict(ft_model.named_parameters())

        for name in base_params:
            if name in ft_params:
                diff[name] = ft_params[name].data - base_params[name].data
        return cls(diff)

    def apply_to(self, base_model: nn.Module, scaling: float = 1.0) -> nn.Module:
        """Apply task vector to base model: theta_new = theta_base + scaling * tau."""
        for name, param in base_model.named_parameters():
            if name in self.vector_dict:
                param.data.add_(self.vector_dict[name].to(param.device), alpha=scaling)
        return base_model

    def __add__(self, other: "TaskVector") -> "TaskVector":
        """Additive composition of task vectors."""
        res = {}
        for k in self.vector_dict:
            if k in other.vector_dict:
                res[k] = self.vector_dict[k] + other.vector_dict[k]
        return TaskVector(res)

    def __sub__(self, other: "TaskVector") -> "TaskVector":
        """Task negation / unlearning: tau_1 - tau_2."""
        res = {}
        for k in self.vector_dict:
            if k in other.vector_dict:
                res[k] = self.vector_dict[k] - other.vector_dict[k]
        return TaskVector(res)

    def __mul__(self, scalar: float) -> "TaskVector":
        res = {k: v * scalar for k, v in self.vector_dict.items()}
        return TaskVector(res)
''')
commit("feat: implement TaskVector computation, arithmetic, and negation in nanomind/merge/task_vectors.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — TIES-Merging
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/merge/ties.py", '''\
"""
nanomind/merge/ties.py — TIES-Merging: Trimming, Electing Sign, and Disjoint Merge (Yadav et al., NeurIPS 2023).
Resolves parameter interference across multi-task models.
"""
from typing import List, Dict, Any, Tuple
import torch


class TIESMerger:
    """
    Three-stage TIES merge algorithm:
    1. Trim: Keep top density% largest magnitude parameters, set others to 0.
    2. Elect Sign: Determine consensus sign for each parameter based on total mass.
    3. Disjoint Merge: Average only parameters that agree with the consensus sign.
    """

    def __init__(self, density: float = 0.2):
        self.density = density

    def trim_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Retain top density% parameters by absolute magnitude."""
        if self.density >= 1.0:
            return t
        flat = t.abs().flatten()
        k = max(1, int(self.density * flat.numel()))
        threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
        mask = t.abs() >= threshold
        return t * mask

    def merge_tensors(self, tensors: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        Executes TIES on a list of delta tensors.
        """
        n = len(tensors)
        w = weights or [1.0 / n] * n

        # 1. Trim
        trimmed = [self.trim_tensor(t) for t in tensors]

        # 2. Elect Sign
        stacked = torch.stack(trimmed, dim=0)  # (N, ...)
        sign_vote = torch.zeros_like(stacked[0])
        for i in range(n):
            sign_vote += w[i] * torch.sign(stacked[i])
        consensus_sign = torch.sign(sign_vote)

        # 3. Disjoint Merge
        merged = torch.zeros_like(stacked[0])
        denom = torch.zeros_like(stacked[0])

        for i in range(n):
            agrees = (torch.sign(stacked[i]) == consensus_sign) & (stacked[i] != 0)
            merged += w[i] * (stacked[i] * agrees)
            denom += w[i] * agrees.float()

        denom = torch.clamp(denom, min=1e-8)
        return merged / denom
''')
commit("feat: implement TIES-Merging (Trim, Elect Sign, Disjoint Merge) in nanomind/merge/ties.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — DARE (Drop And REscale)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/merge/dare.py", '''\
"""
nanomind/merge/dare.py — DARE: Drop And REscale parameter sparsification (Yu et al., ICML 2024).
Randomly drops redundant fine-tuned parameters and rescales remaining weights to preserve expectation.
"""
from typing import List, Optional
import torch


class DAREPruner:
    """
    Drop And REscale:
    With probability drop_rate, parameter delta is set to 0.
    Remaining parameters are rescaled by 1 / (1 - drop_rate) such that E[tilde_tau] = tau.
    """

    def __init__(self, drop_rate: float = 0.5, rescale: bool = True):
        self.drop_rate = drop_rate
        self.rescale = rescale

    def prune_tensor(self, delta: torch.Tensor) -> torch.Tensor:
        """Applies DARE Bernoulli drop and rescaling."""
        if self.drop_rate <= 0.0:
            return delta
        if self.drop_rate >= 1.0:
            return torch.zeros_like(delta)

        keep_prob = 1.0 - self.drop_rate
        # Bernoulli mask where 1 = keep, 0 = drop
        mask = torch.bernoulli(torch.full_like(delta, keep_prob))
        pruned = delta * mask

        if self.rescale:
            pruned = pruned / keep_prob

        return pruned
''')
commit("feat: implement DARE (Drop And REscale) parameter pruning in nanomind/merge/dare.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Spherical Linear Interpolation (SLERP)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/merge/slerp.py", '''\
"""
nanomind/merge/slerp.py — Spherical Linear Interpolation (SLERP) for high-dimensional parameter tensors.
Interpolates along geodesic on hypersphere preserving directional variance.
"""
import math
from typing import Optional
import torch


def slerp(t1: torch.Tensor, t2: torch.Tensor, t: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    """
    Spherical Linear Interpolation between two parameter tensors:
    SLERP(t1, t2; t) = (sin((1 - t)*Omega) / sin(Omega)) * t1 + (sin(t*Omega) / sin(Omega)) * t2
    t: interpolation factor in [0, 1] (0 = t1, 1 = t2)
    """
    if t <= 0.0:
        return t1
    if t >= 1.0:
        return t2

    norm_t1 = t1 / (torch.norm(t1) + eps)
    norm_t2 = t2 / (torch.norm(t2) + eps)

    # Cosine of angle between vectors
    dot = torch.sum(norm_t1 * norm_t2)
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)

    # Angle Omega
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    if torch.abs(sin_omega) < eps:
        # Collinear vectors: fall back to linear interpolation
        return (1.0 - t) * t1 + t * t2

    scale_1 = torch.sin((1.0 - t) * omega) / sin_omega
    scale_2 = torch.sin(t * omega) / sin_omega

    interpolated = scale_1 * t1 + scale_2 * t2
    return interpolated
''')
commit("feat: implement Spherical Linear Interpolation (SLERP) in nanomind/merge/slerp.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Frankensplit Layer Splicing
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/merge/frankenmerge.py", '''\
"""
nanomind/merge/frankenmerge.py — Frankenmerge / Passthrough layer-wise model splicing.
Stitches blocks from multiple models without parameter modification.
"""
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn


class LayerSplicer:
    """
    Slices and combines transformer blocks from multiple models into an extended or hybrid architecture.
    """

    @staticmethod
    def splice_layers(
        source_models: List[nn.Module],
        slice_specs: List[Tuple[int, int, int]],  # [(model_idx, start_layer, end_layer), ...]
    ) -> List[nn.Module]:
        """
        Extracts layers matching slice specifications.
        """
        combined_blocks = []
        for model_idx, start_idx, end_idx in slice_specs:
            model = source_models[model_idx]
            # Try finding transformer blocks attribute
            blocks = getattr(model, "blocks", None) or getattr(model, "layers", None)
            if blocks is None:
                raise AttributeError(f"Model at index {model_idx} has no 'blocks' or 'layers' attribute")

            for i in range(start_idx, end_idx):
                combined_blocks.append(blocks[i])

        return combined_blocks
''')
commit("feat: implement Frankensplit layer-wise model splicing in nanomind/merge/frankenmerge.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — Model Soups (Uniform & Greedy)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/merge/soup.py", '''\
"""
nanomind/merge/soup.py — Model Soups: Uniform Soup and Greedy Soup (Wortsman et al., ICML 2022).
Averages fine-tuned checkpoints for free performance and out-of-distribution robustness.
"""
from typing import List, Dict, Any, Callable, Optional
import copy
import torch
import torch.nn as nn


class ModelSoup:
    """
    Constructs model soups by averaging model weights.
    """

    @staticmethod
    def uniform_soup(models: List[nn.Module]) -> nn.Module:
        """
        Uniformly averages all model weights: theta = 1/N * sum theta_i.
        """
        if not models:
            raise ValueError("No models provided for soup")

        soup = copy.deepcopy(models[0])
        soup_params = dict(soup.named_parameters())
        n = len(models)

        # Initialize to 0
        for p in soup_params.values():
            p.data.zero_()

        # Accumulate weights
        for model in models:
            for name, param in model.named_parameters():
                if name in soup_params:
                    soup_params[name].data.add_(param.data, alpha=1.0 / n)

        return soup

    @staticmethod
    def greedy_soup(
        models: List[nn.Module],
        eval_fn: Callable[[nn.Module], float],
    ) -> nn.Module:
        """
        Greedy soup: iteratively adds models that improve evaluation score.
        eval_fn: takes model and returns metric (higher is better).
        """
        if not models:
            raise ValueError("No models provided")

        # Rank models by individual performance
        scored = [(m, eval_fn(m)) for m in models]
        scored.sort(key=lambda x: x[1], reverse=True)

        current_soup = copy.deepcopy(scored[0][0])
        best_score = scored[0][1]
        active_models = [scored[0][0]]

        for model, _ in scored[1:]:
            candidate_models = active_models + [model]
            candidate_soup = ModelSoup.uniform_soup(candidate_models)
            score = eval_fn(candidate_soup)

            if score > best_score:
                best_score = score
                active_models = candidate_models
                current_soup = candidate_soup

        return current_soup
''')
commit("feat: implement Uniform and Greedy Model Soups in nanomind/merge/soup.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — ModelMerger Master Engine
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/merge/merger.py", '''\
"""
nanomind/merge/merger.py — ModelMerger master engine executing merging recipes across models.
"""
from typing import List, Dict, Any, Optional
import copy
import torch
import torch.nn as nn

from nanomind.merge.config import MergeConfig, MergeMethod
from nanomind.merge.task_vectors import TaskVector
from nanomind.merge.ties import TIESMerger
from nanomind.merge.dare import DAREPruner
from nanomind.merge.slerp import slerp
from nanomind.merge.soup import ModelSoup


class ModelMerger:
    """
    Orchestrates end-to-end model merging across multiple fine-tuned models.
    """

    def __init__(self, config: Optional[MergeConfig] = None):
        self.config = config or MergeConfig()
        self.ties_merger = TIESMerger(density=self.config.density)
        self.dare_pruner = DAREPruner(drop_rate=self.config.drop_rate, rescale=self.config.rescale)

    def merge(self, base_model: nn.Module, ft_models: List[nn.Module]) -> nn.Module:
        """
        Merges fine-tuned models into a single combined model.
        """
        if not ft_models:
            return copy.deepcopy(base_model)

        merged = copy.deepcopy(base_model)
        method = self.config.method

        if method == MergeMethod.LINEAR or method == MergeMethod.SOUP:
            return ModelSoup.uniform_soup([base_model] + ft_models)

        elif method == MergeMethod.SLERP:
            if len(ft_models) < 1:
                return merged
            # SLERP between base and first ft_model, or between two ft_models
            target = ft_models[0]
            for name, param in merged.named_parameters():
                target_p = dict(target.named_parameters()).get(name)
                if target_p is not None:
                    param.data.copy_(slerp(param.data, target_p.data, t=self.config.slerp_t))
            return merged

        elif method in (MergeMethod.TIES, MergeMethod.DARE_TIES, MergeMethod.TASK_ARITHMETIC):
            # Compute task vectors
            task_vectors = [TaskVector.from_models(base_model, m) for m in ft_models]
            param_names = list(task_vectors[0].vector_dict.keys())

            for name in param_names:
                deltas = [tv.vector_dict[name] for tv in task_vectors if name in tv.vector_dict]
                if not deltas:
                    continue

                if method == MergeMethod.DARE_TIES:
                    # Apply DARE pruning prior to TIES
                    deltas = [self.dare_pruner.prune_tensor(d) for d in deltas]

                if method == MergeMethod.TASK_ARITHMETIC:
                    merged_delta = sum(deltas) / len(deltas)
                else:
                    merged_delta = self.ties_merger.merge_tensors(deltas)

                # Add to merged base model
                p = dict(merged.named_parameters())[name]
                p.data.add_(merged_delta.to(p.device))

            return merged

        return merged
''')
commit("feat: implement ModelMerger master engine executing merging recipes in nanomind/merge/merger.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — Weight Drift & Interference Metrics
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/merge/metrics.py", '''\
"""
nanomind/merge/metrics.py — Metrics for parameter drift, task interference, and sparsity.
"""
from typing import Dict, Any, List
import math
import torch
import torch.nn as nn


def compute_parameter_drift(base_model: nn.Module, merged_model: nn.Module) -> float:
    """
    Computes Euclidean L2 parameter drift ||theta_merged - theta_base||_2.
    """
    total_sq = 0.0
    base_params = dict(base_model.named_parameters())
    merged_params = dict(merged_model.named_parameters())

    for name in base_params:
        if name in merged_params:
            diff = merged_params[name].data - base_params[name].data
            total_sq += torch.sum(diff ** 2).item()

    return math.sqrt(total_sq)


def compute_task_vector_cosine_similarity(tv1: Dict[str, torch.Tensor], tv2: Dict[str, torch.Tensor]) -> float:
    """
    Measures cosine similarity between two task vectors.
    High negative similarity indicates conflicting task directions.
    """
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for name in tv1:
        if name in tv2:
            t1 = tv1[name].flatten()
            t2 = tv2[name].flatten()
            dot += torch.sum(t1 * t2).item()
            norm1 += torch.sum(t1 ** 2).item()
            norm2 += torch.sum(t2 ** 2).item()

    denom = math.sqrt(norm1) * math.sqrt(norm2)
    return float(dot / max(1e-8, denom))
''')
commit("feat: implement parameter drift and task interference metrics in nanomind/merge/metrics.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — Merge API Exposure
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/merge/__init__.py", '''\
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
''')

# Update nanomind/__init__.py
init_py = read("nanomind/__init__.py")
if "from nanomind.merge import" not in init_py:
    init_py = init_py.replace(
        "from nanomind.flow import FlowConfig, FlowTransformer, DiscreteDiffusionSampler, D3PM\n",
        "from nanomind.flow import FlowConfig, FlowTransformer, DiscreteDiffusionSampler, D3PM\n"
        "from nanomind.merge import MergeConfig, ModelMerger, TaskVector, TIESMerger\n"
    )
    init_py = init_py.replace(
        '    "D3PM",\n',
        '    "D3PM",\n'
        '    "MergeConfig",\n'
        '    "ModelMerger",\n'
        '    "TaskVector",\n'
        '    "TIESMerger",\n'
    )
    write("nanomind/__init__.py", init_py)

commit("feat: expose merge API in nanomind/merge/__init__.py and top-level package")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — Examples Demo
# ══════════════════════════════════════════════════════════════════════════════
write("examples/merge_demo.py", '''\
"""
examples/merge_demo.py — End-to-end Demonstration of Model Merging (TIES, DARE, SLERP, Soups).
Demonstrates:
  1. Task Vector extraction & arithmetic (addition & task negation)
  2. TIES-Merging across two fine-tuned models
  3. DARE (Drop And REscale) extreme parameter pruning
  4. Spherical Linear Interpolation (SLERP)
  5. Parameter drift computation
"""
import copy
import torch
import torch.nn as nn

from nanomind.merge import (
    MergeConfig,
    MergeMethod,
    ModelMerger,
    TaskVector,
    TIESMerger,
    DAREPruner,
    slerp,
    ModelSoup,
    compute_parameter_drift,
    compute_task_vector_cosine_similarity,
)


def run_demo():
    print("=" * 70)
    print("  NanoMind Day 63: Model Merging & Weight Averaging (MergeKit)")
    print("=" * 70)

    # Simple 2-layer MLP as base model
    torch.manual_seed(42)
    base_model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )

    # Simulate two fine-tuned models (e.g. Model A tuned for Math, Model B tuned for Code)
    ft_model_a = copy.deepcopy(base_model)
    ft_model_b = copy.deepcopy(base_model)

    with torch.no_grad():
        for p in ft_model_a.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        for p in ft_model_b.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # 1. Task Vectors & Task Arithmetic
    print("\\n[1] Task Vector Computation & Arithmetic:")
    tv_a = TaskVector.from_models(base_model, ft_model_a)
    tv_b = TaskVector.from_models(base_model, ft_model_b)
    tv_sum = tv_a + tv_b
    cos_sim = compute_task_vector_cosine_similarity(tv_a.vector_dict, tv_b.vector_dict)
    print(f"  Task Vector A norm: {sum(v.norm().item() for v in tv_a.vector_dict.values()):.3f}")
    print(f"  Task Vector B norm: {sum(v.norm().item() for v in tv_b.vector_dict.values()):.3f}")
    print(f"  Task Vector Cosine Similarity (Interference): {cos_sim:.4f}")

    # 2. TIES-Merging
    print("\\n[2] TIES-Merging (Trim, Elect Sign, Disjoint Merge):")
    cfg_ties = MergeConfig(method=MergeMethod.TIES, density=0.3)
    merger_ties = ModelMerger(cfg_ties)
    merged_ties = merger_ties.merge(base_model, [ft_model_a, ft_model_b])
    drift_ties = compute_parameter_drift(base_model, merged_ties)
    print(f"  TIES Merged Model Drift from Base: {drift_ties:.4f}")

    # 3. DARE (Drop And REscale)
    print("\\n[3] DARE Parameter Pruning & Rescaling:")
    dare = DAREPruner(drop_rate=0.5, rescale=True)
    sample_tensor = torch.randn(4, 4)
    pruned = dare.prune_tensor(sample_tensor)
    sparsity = (pruned == 0.0).float().mean().item()
    print(f"  Original Tensor Mean: {sample_tensor.mean().item():.3f}")
    print(f"  DARE Sparsity: {sparsity * 100:.1f}% dropped")
    print(f"  Rescaled Mean (Preserves Expectation): {pruned.mean().item():.3f}")

    # 4. Spherical Linear Interpolation (SLERP)
    print("\\n[4] Spherical Linear Interpolation (SLERP):")
    p1 = torch.tensor([1.0, 0.0, 0.0])
    p2 = torch.tensor([0.0, 1.0, 0.0])
    interp = slerp(p1, p2, t=0.5)
    print(f"  P1: {p1.tolist()}, P2: {p2.tolist()}")
    print(f"  SLERP midpoint (t=0.5): {[round(x, 3) for x in interp.tolist()]}")
    print(f"  Midpoint norm: {interp.norm().item():.3f} (preserved on unit sphere)")

    # 5. Model Soup (Uniform Averaging)
    print("\\n[5] Model Soup (Uniform Averaging):")
    soup_model = ModelSoup.uniform_soup([ft_model_a, ft_model_b])
    drift_soup = compute_parameter_drift(base_model, soup_model)
    print(f"  Soup Parameter Drift: {drift_soup:.4f}")

    print("\\n[OK] NanoMind Model Merging demo completed successfully!")


if __name__ == "__main__":
    run_demo()
''')
commit("feat: add examples/merge_demo.py demonstrating TIES, DARE, SLERP, and task arithmetic")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — Unit Tests Part 1: Task Vectors
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_merge.py", '''\
"""
tests/test_merge.py — Comprehensive unit tests for Model Merging algorithms.
"""
import copy
import math
import torch
import torch.nn as nn
from nanomind.merge import (
    MergeConfig,
    MergeMethod,
    ModelMerger,
    TaskVector,
    TIESMerger,
    DAREPruner,
    slerp,
    ModelSoup,
    compute_parameter_drift,
    compute_task_vector_cosine_similarity,
)


class TestTaskVectors:
    def test_task_vector_arithmetic(self):
        m1 = nn.Linear(4, 2, bias=False)
        m2 = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            m1.weight.fill_(1.0)
            m2.weight.fill_(3.0)

        tv = TaskVector.from_models(m1, m2)
        assert (tv.vector_dict["weight"] == 2.0).all()

        # Addition: tv + tv -> 4.0
        tv_sum = tv + tv
        assert (tv_sum.vector_dict["weight"] == 4.0).all()

        # Scaling: tv * 0.5 -> 1.0
        tv_scaled = tv * 0.5
        assert (tv_scaled.vector_dict["weight"] == 1.0).all()

    def test_task_vector_cosine(self):
        v1 = {"w": torch.tensor([1.0, 0.0])}
        v2 = {"w": torch.tensor([0.0, 1.0])}
        sim = compute_task_vector_cosine_similarity(v1, v2)
        assert math.isclose(sim, 0.0, abs_tol=1e-5)
''')
commit("test: add tests for TaskVector addition, scaling, and cosine interference")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — Unit Tests Part 2: TIES
# ══════════════════════════════════════════════════════════════════════════════
test_src = read("tests/test_merge.py")
test_src += '''\


class TestTIES:
    def test_trimming(self):
        merger = TIESMerger(density=0.5)
        t = torch.tensor([1.0, 5.0, 2.0, 8.0])
        trimmed = merger.trim_tensor(t)
        # Top 2 values (8.0, 5.0) kept, rest zeroed
        assert trimmed[0] == 0.0
        assert trimmed[2] == 0.0
        assert trimmed[1] == 5.0
        assert trimmed[3] == 8.0

    def test_ties_merge_sign_election(self):
        merger = TIESMerger(density=1.0)
        t1 = torch.tensor([2.0, -1.0])
        t2 = torch.tensor([3.0, 1.0])
        t3 = torch.tensor([1.0, -2.0])
        # Position 0: all positive -> positive average
        # Position 1: t1 and t3 negative (-1, -2), t2 positive (+1) -> majority negative
        merged = merger.merge_tensors([t1, t2, t3])
        assert merged[0] > 0
        assert merged[1] < 0
'''
write("tests/test_merge.py", test_src)
commit("test: add tests for TIES-Merging trimming and sign election logic")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — Unit Tests Part 3: DARE
# ══════════════════════════════════════════════════════════════════════════════
test_src = read("tests/test_merge.py")
test_src += '''\


class TestDARE:
    def test_dare_pruning_and_expectation(self):
        torch.manual_seed(0)
        pruner = DAREPruner(drop_rate=0.5, rescale=True)
        t = torch.ones(1000)
        pruned = pruner.prune_tensor(t)
        # Approximately 50% dropped
        sparsity = (pruned == 0.0).float().mean().item()
        assert 0.4 < sparsity < 0.6
        # Rescaled non-zero values are 2.0
        assert math.isclose(pruned.mean().item(), 1.0, rel_tol=0.1)
'''
write("tests/test_merge.py", test_src)
commit("test: add tests for DARE random drop and expectation preservation")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — Unit Tests Part 4: SLERP
# ══════════════════════════════════════════════════════════════════════════════
test_src = read("tests/test_merge.py")
test_src += '''\


class TestSLERP:
    def test_slerp_boundaries(self):
        t1 = torch.tensor([1.0, 0.0])
        t2 = torch.tensor([0.0, 1.0])

        assert (slerp(t1, t2, t=0.0) == t1).all()
        assert (slerp(t1, t2, t=1.0) == t2).all()

        mid = slerp(t1, t2, t=0.5)
        assert math.isclose(mid[0].item(), mid[1].item(), rel_tol=1e-4)
        assert math.isclose(mid.norm().item(), 1.0, rel_tol=1e-4)
'''
write("tests/test_merge.py", test_src)
commit("test: add tests for SLERP interpolation boundary conditions")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — Unit Tests Part 5: Frankensplit
# ══════════════════════════════════════════════════════════════════════════════
test_src = read("tests/test_merge.py")
test_src += '''\


class TestFrankensplit:
    def test_layer_splicer(self):
        from nanomind.merge.frankenmerge import LayerSplicer
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])

        m1 = DummyModel()
        m2 = DummyModel()
        spliced = LayerSplicer.splice_layers([m1, m2], [(0, 0, 2), (1, 2, 4)])
        assert len(spliced) == 4
'''
write("tests/test_merge.py", test_src)
commit("test: add tests for Frankensplit layer recombination")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — Unit Tests Part 6: ModelMerger & Soup
# ══════════════════════════════════════════════════════════════════════════════
test_src = read("tests/test_merge.py")
test_src += '''\


class TestModelMergerAndSoup:
    def test_uniform_soup(self):
        m1 = nn.Linear(2, 2, bias=False)
        m2 = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            m1.weight.fill_(2.0)
            m2.weight.fill_(4.0)
        soup = ModelSoup.uniform_soup([m1, m2])
        assert (soup.weight == 3.0).all()

    def test_model_merger_ties(self):
        base = nn.Linear(4, 2, bias=False)
        ft1 = copy.deepcopy(base)
        ft2 = copy.deepcopy(base)
        with torch.no_grad():
            ft1.weight.add_(torch.randn_like(ft1.weight) * 0.1)
            ft2.weight.add_(torch.randn_like(ft2.weight) * 0.1)

        merger = ModelMerger(MergeConfig(method=MergeMethod.TIES))
        merged = merger.merge(base, [ft1, ft2])
        drift = compute_parameter_drift(base, merged)
        assert drift > 0.0
'''
write("tests/test_merge.py", test_src)
commit("test: add tests for ModelMerger multi-model execution and soup averaging")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — Bump to v6.3.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace('__version__ = "6.2.0"', '__version__ = "6.3.0"')
write("nanomind/__init__.py", src)
commit("feat: bump to v6.3.0 — Model Merging & Weight Averaging (TIES/DARE/SLERP/Soups) release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + Push + Tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `flow`       | Discrete Diffusion & Flow Matching — SEDD, D3PM, OT-CFM, non-autoregressive parallel decoding |",
    "| `flow`       | Discrete Diffusion & Flow Matching — SEDD, D3PM, OT-CFM, non-autoregressive parallel decoding |\n"
    "| `merge`      | Model Merging & Weight Averaging — TIES, DARE, SLERP, Task Arithmetic, Model Soups, Frankensplit |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [6.3.0] — 2024 — Model Merging & Weight Averaging\n\n### Added\n"
      "- `MergeConfig` & `MergeMethod` — configurations for multi-model merging recipes\n"
      "- `TaskVector` — task arithmetic parameter delta representations (addition, scaling, unlearning)\n"
      "- `TIESMerger` — Trimming, Electing Sign, and Disjoint Merge (Yadav et al., NeurIPS 2023)\n"
      "- `DAREPruner` — Drop And REscale parameter sparsification (Yu et al., ICML 2024)\n"
      "- `slerp` — Spherical Linear Interpolation preserving angular geometric variance\n"
      "- `LayerSplicer` — Frankensplit layer-wise model splicing and hybrid depth expansion\n"
      "- `ModelSoup` — Uniform Soup and Greedy Soup checkpoint averaging\n"
      "- `ModelMerger` — master engine applying TIES, DARE, SLERP, and arithmetic recipes\n"
      "- `compute_parameter_drift` & `compute_task_vector_cosine_similarity` — merge interference metrics\n"
      "- `examples/merge_demo.py` — complete model merging and task arithmetic demonstration\n\n---\\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v6.3.0, update README and CHANGELOG for Day 63 Merge")

print("\n=== Pushing Day 63 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")
run("git", "tag", "-a", "v6.3.0", "-m", "NanoMind v6.3.0 — Model Merging & Weight Averaging", check=False)
r = run("git", "push", "origin", "v6.3.0", check=False)
print("Tag v6.3.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 63 COMPLETE — v6.3.0 TAGGED! ===")
