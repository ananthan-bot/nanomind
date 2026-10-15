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
