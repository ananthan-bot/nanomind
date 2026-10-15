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
