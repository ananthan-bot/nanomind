"""
nanomind/omni/generator.py — Multi-head streaming generation for text, speech tokens, and tools.
"""
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.omni.config import OmniConfig


class OmniGenerator(nn.Module):
    """
    Decodes representations into synchronized dual modalities:
    1. Text vocabulary head
    2. Speech RVQ acoustic code head
    3. Tool calling classification
    """

    def __init__(self, config: Optional[OmniConfig] = None):
        super().__init__()
        self.config = config or OmniConfig()
        d_model = self.config.d_model

        # Output projection heads
        self.text_head = nn.Linear(d_model, self.config.text_vocab_size, bias=False)
        self.speech_head = nn.Linear(d_model, self.config.audio_codebook_size, bias=False)
        self.tool_trigger_head = nn.Linear(d_model, 2)  # Binary classification: [normal_token, tool_call_trigger]

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        hidden_states: (B, T, D)
        Returns:
            dict of logits: text, speech, tool_trigger
        """
        text_logits = self.text_head(hidden_states)
        speech_logits = self.speech_head(hidden_states)
        tool_logits = self.tool_trigger_head(hidden_states)

        return {
            "text_logits": text_logits,
            "speech_logits": speech_logits,
            "tool_logits": tool_logits,
        }
