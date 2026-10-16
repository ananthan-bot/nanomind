"""
nanomind/grammar/sampler.py — High-level constrained sampling loop with model logit masking.
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Callable, Tuple
from nanomind.grammar.masker import LogitGrammarMasker
from nanomind.grammar.validator import StreamingGrammarValidator


class ConstrainedSampler:
    """
    Executes model generation with grammar-constrained logit masking,
    ensuring 100% syntactically valid structured generation.
    """
    def __init__(
        self,
        vocab: Dict[int, str],
        fill_value: float = -1e9,
    ):
        self.vocab = vocab
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.masker = LogitGrammarMasker(vocab_size=len(vocab), fill_value=fill_value)

    def sample_step(
        self,
        logits: torch.Tensor,
        valid_token_ids: List[int],
        temperature: float = 1.0,
    ) -> int:
        """Apply grammar mask and sample the next token."""
        masked_logits = self.masker.mask_logits(logits, valid_token_ids)
        if temperature > 0:
            probs = F.softmax(masked_logits / temperature, dim=-1)
            token_id = int(torch.multinomial(probs, num_samples=1).item())
        else:
            token_id = int(torch.argmax(masked_logits, dim=-1).item())
        return token_id

    def generate_constrained(
        self,
        model_forward_fn: Callable[[List[int]], torch.Tensor],
        validator_filter_fn: Callable[[str], List[int]],
        initial_token_ids: List[int],
        max_new_tokens: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[List[int], str]:
        """Generate tokens until completion or max_new_tokens while respecting grammar filter."""
        generated_ids = list(initial_token_ids)
        validator = StreamingGrammarValidator()
        current_text = "".join(self.vocab.get(tid, "") for tid in initial_token_ids)

        for _ in range(max_new_tokens):
            logits = model_forward_fn(generated_ids)
            valid_tokens = validator_filter_fn(current_text)
            if not valid_tokens:
                break
            next_id = self.sample_step(logits, valid_tokens, temperature=0.0)
            token_str = self.vocab.get(next_id, "")
            current_text += token_str
            generated_ids.append(next_id)
            validator.feed_token(token_str)

            if eos_token_id is not None and next_id == eos_token_id:
                break
            if validator.is_complete_json():
                break

        return generated_ids, current_text
