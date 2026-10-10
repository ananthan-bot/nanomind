"""
nanomind/omni/loss.py — Joint multi-task omni loss: Text CE + Speech Codebook CE + Cross-Modal Alignment.
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class OmniMultiTaskLoss(nn.Module):
    """
    Joint loss function for unified omni-modal training:
    L_total = L_text + lambda_speech * L_speech + lambda_tool * L_tool
    """

    def __init__(self, lambda_speech: float = 1.0, lambda_tool: float = 0.5):
        super().__init__()
        self.lambda_speech = lambda_speech
        self.lambda_tool = lambda_tool

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        text_targets: Optional[torch.Tensor] = None,
        speech_targets: Optional[torch.Tensor] = None,
        tool_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_text = torch.tensor(0.0, device=predictions["hidden_states"].device)
        loss_speech = torch.tensor(0.0, device=predictions["hidden_states"].device)
        loss_tool = torch.tensor(0.0, device=predictions["hidden_states"].device)

        if text_targets is not None:
            text_logits = predictions["text_logits"]
            loss_text = F.cross_entropy(text_logits.reshape(-1, text_logits.size(-1)), text_targets.reshape(-1))

        if speech_targets is not None:
            speech_logits = predictions["speech_logits"]
            loss_speech = F.cross_entropy(speech_logits.reshape(-1, speech_logits.size(-1)), speech_targets.reshape(-1))

        if tool_targets is not None:
            tool_logits = predictions["tool_logits"]
            loss_tool = F.cross_entropy(tool_logits.reshape(-1, tool_logits.size(-1)), tool_targets.reshape(-1))

        total_loss = loss_text + self.lambda_speech * loss_speech + self.lambda_tool * loss_tool

        return {
            "total_loss": total_loss,
            "loss_text": loss_text,
            "loss_speech": loss_speech,
            "loss_tool": loss_tool,
        }
