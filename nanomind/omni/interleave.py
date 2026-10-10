"""
nanomind/omni/interleave.py — Any-to-any multimodal sequence interleaving and modality masks.
"""
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from nanomind.omni.config import ModalityType


class MultimodalItem:
    """An atomic multimodal item: text string, image tensor, audio tensor, or tool call."""

    def __init__(self, modality: ModalityType, data: Any):
        self.modality = modality
        self.data = data

    def __repr__(self) -> str:
        return f"MultimodalItem(modality={self.modality.value})"


class MultimodalSequenceBuilder:
    """
    Constructs interleaved sequences combining text, vision, and audio tokens with modality indicators.
    """

    MODALITY_MAP = {
        ModalityType.TEXT: 0,
        ModalityType.IMAGE: 1,
        ModalityType.AUDIO: 2,
        ModalityType.TOOL: 3,
    }

    def __init__(self, d_model: int = 256):
        self.d_model = d_model

    def build_sequence(
        self,
        token_tensors: List[Tuple[ModalityType, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        token_tensors: list of (ModalityType, tensor of shape (B, T_i, D))
        Returns:
            interleaved_embeds: (B, T_total, D)
            modality_ids: (B, T_total) integer tensor of modality indicators
        """
        if not token_tensors:
            raise ValueError("Empty multimodal sequence")

        embed_list = []
        id_list = []

        for mod_type, t in token_tensors:
            embed_list.append(t)
            B, T_len, _ = t.shape
            mod_val = self.MODALITY_MAP[mod_type]
            mod_id_tensor = torch.full((B, T_len), mod_val, dtype=torch.long, device=t.device)
            id_list.append(mod_id_tensor)

        interleaved_embeds = torch.cat(embed_list, dim=1)
        modality_ids = torch.cat(id_list, dim=1)

        return interleaved_embeds, modality_ids
