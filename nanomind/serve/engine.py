"""
nanomind/serve/engine.py — Inference engine wrapping model + tokenizer.
"""

from __future__ import annotations

import time
import torch
import torch.nn.functional as F

from nanomind.tokenizer.base import BaseTokenizer
from nanomind.serve.schemas import (
    GenerateRequest, GenerateResponse, TokenizeRequest, TokenizeResponse
)
from nanomind.utils.logger import get_logger

log = get_logger("serve.engine")


def _sample(
    logits:      torch.Tensor,
    temperature: float = 1.0,
    top_k:       int   = 0,
    top_p:       float = 1.0,
) -> int:
    """Sample next token from logits with temperature/top-K/top-P."""
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        topk_vals = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
        logits     = logits.masked_fill(logits < topk_vals, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = sorted_probs.cumsum(0)
        remove = cum - sorted_probs > top_p
        sorted_probs[remove] = 0.0
        sorted_probs /= sorted_probs.sum()
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
    return torch.multinomial(probs, 1).item()


class InferenceEngine:
    """
    Core inference engine: wraps model + tokenizer for the server.

    Handles:
    - Greedy / sampled token generation
    - Stop string detection
    - Token throughput tracking

    Args:
        model:     Language model with a standard forward() interface.
        tokenizer: Tokenizer for encoding prompts and decoding output.
        device:    Inference device.
        model_name: Human-readable name shown in /info.
    """

    def __init__(
        self,
        model:      torch.nn.Module,
        tokenizer:  BaseTokenizer,
        device:     str | torch.device = "cpu",
        model_name: str = "NanoMind",
    ) -> None:
        self.model      = model.eval()
        self.tokenizer  = tokenizer
        self.device     = torch.device(device)
        self.model_name = model_name
        self._total_tokens = 0
        self._total_reqs   = 0
        log.info(f"InferenceEngine ready on {self.device}")

    @torch.no_grad()
    def generate(self, req: GenerateRequest) -> GenerateResponse:
        """
        Generate text for a single request.

        Args:
            req: GenerateRequest with prompt and sampling parameters.

        Returns:
            GenerateResponse with generated text and token counts.
        """
        t0      = time.perf_counter()
        enc     = self.tokenizer.encode(req.prompt)
        ids     = torch.tensor([enc], dtype=torch.long, device=self.device)
        n_prompt = len(enc)
        generated: list[int] = []
        finish_reason = "length"

        for _ in range(req.max_new_tokens):
            # Truncate to block_size if model has one
            block = getattr(self.model, "cfg", None)
            if block and hasattr(block, "block_size"):
                ids = ids[:, -block.block_size:]

            logits, _ = self.model(ids)
            next_tok  = _sample(logits[0, -1, :], req.temperature, req.top_k, req.top_p)
            generated.append(next_tok)
            ids       = torch.cat([ids, torch.tensor([[next_tok]], device=self.device)], dim=1)

            # Check stop strings
            current_text = self.tokenizer.decode(generated)
            for stop in req.stop:
                if stop in current_text:
                    finish_reason = "stop"
                    break
            if finish_reason == "stop":
                break

        text  = self.tokenizer.decode(generated)
        elapsed = time.perf_counter() - t0
        self._total_tokens += len(generated)
        self._total_reqs   += 1
        log.debug(f"Generated {len(generated)} tokens in {elapsed*1000:.1f}ms")

        return GenerateResponse(
            text=text,
            prompt_tokens=n_prompt,
            generated_tokens=len(generated),
            finish_reason=finish_reason,
            request_id=req.request_id,
            model=self.model_name,
        )

    def tokenize(self, req: TokenizeRequest) -> TokenizeResponse:
        """Tokenize text and return token IDs."""
        tokens = self.tokenizer.encode(req.text)
        return TokenizeResponse(
            tokens=tokens,
            n_tokens=len(tokens),
            text=self.tokenizer.decode(tokens),
        )

    def info(self) -> dict:
        """Return model metadata as a dict."""
        cfg = getattr(self.model, "cfg", None)
        n   = sum(p.numel() for p in self.model.parameters())
        return {
            "model":      self.model_name,
            "n_params":   n,
            "vocab_size": getattr(cfg, "vocab_size", -1) if cfg else -1,
            "d_model":    getattr(cfg, "d_model",    -1) if cfg else -1,
            "n_layers":   getattr(cfg, "n_layers",   -1) if cfg else -1,
            "n_heads":    getattr(cfg, "n_heads",    -1) if cfg else -1,
            "block_size": getattr(cfg, "block_size", -1) if cfg else -1,
            "device":     str(self.device),
            "total_reqs":    self._total_reqs,
            "total_tokens":  self._total_tokens,
        }
