"""
nanomind/omni/pipeline.py — High-level user interface pipeline for unified multimodal chat.
"""
from typing import Optional, Dict, Any, List
import torch

from nanomind.omni.model import NanoMindOmni
from nanomind.omni.duplex import DuplexDialogueManager


class OmniResponse:
    """Container for multimodal outputs."""

    def __init__(self, text_token_ids: List[int], speech_token_ids: List[int], has_tool_trigger: bool = False):
        self.text_token_ids = text_token_ids
        self.speech_token_ids = speech_token_ids
        self.has_tool_trigger = has_tool_trigger

    def __repr__(self) -> str:
        return f"OmniResponse(tokens={len(self.text_token_ids)}, speech_codes={len(self.speech_token_ids)})"


class OmniPipeline:
    """
    Unified high-level conversation pipeline supporting text, vision, and speech.
    """

    def __init__(self, model: NanoMindOmni):
        self.model = model
        self.duplex_manager = DuplexDialogueManager()

    def chat(
        self,
        text_tokens: Optional[torch.Tensor] = None,
        image_patches: Optional[torch.Tensor] = None,
        audio_frames: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
    ) -> OmniResponse:
        """
        Interactive multimodal chat method.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                text_tokens=text_tokens,
                image_patches=image_patches,
                audio_frames=audio_frames,
            )
            # Greedy prediction on latest token position
            pred_text = int(torch.argmax(outputs["text_logits"][:, -1, :]).item())
            pred_audio = int(torch.argmax(outputs["speech_logits"][:, -1, :]).item())
            is_tool = bool(torch.argmax(outputs["tool_logits"][:, -1, :]).item() == 1)

            return OmniResponse(
                text_token_ids=[pred_text],
                speech_token_ids=[pred_audio],
                has_tool_trigger=is_tool,
            )
