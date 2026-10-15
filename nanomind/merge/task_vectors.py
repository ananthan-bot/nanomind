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
