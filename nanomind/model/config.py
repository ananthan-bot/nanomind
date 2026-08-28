"""
nanomind/model/config.py — NanoMind model configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Full configuration for a NanoMind transformer LLM.

    Attributes:
        vocab_size:     Size of the token vocabulary.
        block_size:     Maximum context window length (sequence length).
        d_model:        Embedding dimension.
        n_layers:       Number of stacked transformer blocks.
        n_heads:        Number of attention heads per block.
        d_ff:           FFN hidden dimension. None defaults to 4 * d_model.
        dropout:        Dropout probability (set to 0 for inference).
        norm_type:      Normalization — ``"layernorm"`` or ``"rmsnorm"``.
        activation:     FFN activation — ``"gelu"`` or ``"swiglu"``.
        norm_placement: ``"pre"`` (Pre-LN) or ``"post"`` (Post-LN).
        bias:           Whether to use bias in attention projections.
        weight_tying:   Tie token embedding weights to LM head.
    """

    vocab_size:     int       = 256
    block_size:     int       = 128
    d_model:        int       = 128
    n_layers:       int       = 4
    n_heads:        int       = 4
    d_ff:           int | None = None
    dropout:        float     = 0.1
    norm_type:      str       = "layernorm"
    activation:     str       = "gelu"
    norm_placement: str       = "pre"
    bias:           bool      = False
    weight_tying:   bool      = True
    pos_type:       str       = "learned"  # "learned", "rope", "alibi"
    n_kv_heads:     int | None = None        # None = same as n_heads (standard MHA)
    window_size:    int | None = None        # None = full attention; int = SWA window

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.n_layers > 0,   "n_layers must be positive"
        assert self.block_size > 0, "block_size must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.pos_type in ("learned", "rope", "alibi")
        if self.n_kv_heads is not None:
            assert self.n_heads % self.n_kv_heads == 0, (
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        assert self.norm_type in ("layernorm", "rmsnorm")
        assert self.activation in ("gelu", "swiglu")
        assert self.norm_placement in ("pre", "post")

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.n_heads

    @property
    def effective_d_ff(self) -> int:
        """Resolved FFN hidden dimension."""
        return self.d_ff or 4 * self.d_model

    def to_dict(self) -> dict:
        """Serialize config to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Deserialize config from a plain dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save_json(self, path: str | Path) -> None:
        """Save config to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        """Load config from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
