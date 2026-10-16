"""
nanomind/grammar/masker.py — Dynamic logit masking engine based on valid grammar transitions.
"""
import torch
from typing import List, Set, Dict, Optional


class LogitGrammarMasker:
    """
    Filters model logits by setting forbidden token logits to negative infinity,
    guaranteeing only syntactically permitted tokens are sampled.
    """
    def __init__(self, vocab_size: int, fill_value: float = -1e9):
        self.vocab_size = vocab_size
        self.fill_value = fill_value

    def mask_logits(self, logits: torch.Tensor, valid_token_ids: List[int]) -> torch.Tensor:
        """
        Apply grammar mask to logits tensor.
        logits: Tensor of shape (..., vocab_size)
        valid_token_ids: list of integer token IDs allowed by grammar
        """
        if not valid_token_ids:
            return logits

        masked = torch.full_like(logits, self.fill_value)
        # Handle 1D or 2D logits
        if logits.dim() == 1:
            masked[valid_token_ids] = logits[valid_token_ids]
        elif logits.dim() == 2:
            batch_size = logits.shape[0]
            for b in range(batch_size):
                masked[b, valid_token_ids] = logits[b, valid_token_ids]
        return masked

    def create_mask_tensor(self, valid_token_ids: List[int], device: torch.device = None) -> torch.Tensor:
        """Create a boolean mask tensor where True indicates allowed tokens."""
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        mask[valid_token_ids] = True
        return mask
