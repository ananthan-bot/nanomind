"""
nanomind/edit/editor.py — ModelEditor master interface executing single and batch factual edits.
"""
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn

from nanomind.edit.config import EditConfig, KnowledgeFact
from nanomind.edit.rome import ROMEEngine
from nanomind.edit.memit import MEMITEngine
from nanomind.edit.covariance import CovarianceCollector
from nanomind.edit.kv_extract import KeyValueExtractor


class ModelEditor:
    """
    High-level editor coordinating ROME and MEMIT updates across model layers.
    """

    def __init__(self, config: Optional[EditConfig] = None):
        self.config = config or EditConfig()
        self.rome = ROMEEngine(self.config)
        self.memit = MEMITEngine(self.config)

    def edit_single_fact(
        self,
        layer: nn.Linear,
        key: torch.Tensor,
        target_value: torch.Tensor,
        cov_collector: CovarianceCollector,
    ):
        """Executes ROME rank-one edit on a single linear projection layer."""
        c_inv = cov_collector.get_inverse()
        self.rome.apply_edit(layer, key, target_value, c_inv)

    def edit_batch_facts(
        self,
        layer: nn.Linear,
        keys: torch.Tensor,
        target_values: torch.Tensor,
        cov_collector: CovarianceCollector,
    ):
        """Executes MEMIT batch edit across M facts simultaneously."""
        with torch.no_grad():
            delta_w = self.memit.compute_batch_update(
                weight=layer.weight.data,
                keys=keys,
                values=target_values,
                c_matrix=cov_collector.cov_matrix,
            )
            layer.weight.data.add_(delta_w)
