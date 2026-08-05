"""
model.py — NanoMind: a small GPT-style transformer language model

Architecture:
  - Token + learned positional embeddings
  - N x TransformerBlock (pre-norm)
      └─ CausalSelfAttention (multi-head, masked)
      └─ FeedForward (2-layer MLP, GELU)
  - Final LayerNorm
  - Linear LM head (tied to token embedding weights)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


# ---------------------------------------------------------------------------
# Causal Self-Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    The causal mask prevents each position from attending to future positions,
    which is essential for autoregressive language modelling.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model

        # Fused QKV projection
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        # Output projection
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Register the causal mask as a non-parameter buffer
        T = cfg.block_size
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, T, T))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, time, channels

        # Compute Q, K, V and split into heads
        qkv = self.qkv(x)                                      # (B, T, 3*C)
        q, k, v = qkv.split(self.d_model, dim=-1)              # each (B, T, C)

        # Reshape to (B, n_heads, T, head_dim)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention with causal mask
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale               # (B, H, T, T)
        attn = attn.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate values
        y = attn @ v                                            # (B, H, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)       # (B, T, C)

        return self.resid_drop(self.out_proj(y))


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Expands to 4x the model dimension, applies GELU, then projects back.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = 4 * cfg.d_model
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    A single transformer block using Pre-Norm (LayerNorm before each sub-layer).

    Pre-Norm: x = x + Sublayer(LN(x))
    This is more stable than Post-Norm for training deep models.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# MiniGPT
# ---------------------------------------------------------------------------

class MiniGPT(nn.Module):
    """
    NanoMind — a small GPT-style causal language model.

    Forward pass: token_ids (B, T) → logits (B, T, vocab_size)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # LM head — tied to token embedding weights (weight tying saves params)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        # Initialise weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layers) for stability
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "net.2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx:     (B, T) integer token IDs
            targets: (B, T) integer token IDs shifted by one (optional, for loss)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss (or None if targets not provided)
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"Sequence length {T} exceeds block_size {self.cfg.block_size}"
        )

        positions = torch.arange(T, device=idx.device)                  # (T,)
        x = self.emb_drop(self.token_emb(idx) + self.pos_emb(positions))  # (B, T, d_model)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                         # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Text Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressively generate tokens appended to `idx`.

        Args:
            idx:            (1, T) seed token IDs
            max_new_tokens: number of tokens to generate
            temperature:    > 1 = more random, < 1 = more focused
            top_k:          if set, sample only from top-k logits

        Returns:
            (1, T + max_new_tokens) tensor of token IDs
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.cfg.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature                      # (1, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)           # (1, 1)
            idx = torch.cat([idx, next_tok], dim=1)

        return idx

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)

    def __repr__(self) -> str:
        n = self.num_parameters()
        return (
            f"NanoMind("
            f"vocab={self.cfg.vocab_size}, "
            f"d_model={self.cfg.d_model}, "
            f"layers={self.cfg.n_layers}, "
            f"heads={self.cfg.n_heads}, "
            f"params={n:,})"
        )
