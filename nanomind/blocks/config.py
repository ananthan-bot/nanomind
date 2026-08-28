"""
nanomind/blocks/config.py — Configuration for a single Transformer block.
"""

from dataclasses import dataclass


@dataclass
class BlockConfig:
    """
    Configuration for one :class:`~nanomind.blocks.TransformerBlock`.

    Attributes:
        d_model:        Embedding dimension.
        n_heads:        Number of attention heads.
        block_size:     Maximum sequence length.
        d_ff:           FFN hidden dimension. Defaults to 4 * d_model if None.
        dropout:        Dropout probability.
        norm_type:      ``"layernorm"`` (default) or ``"rmsnorm"``.
        activation:     FFN activation — ``"gelu"`` (default) or ``"swiglu"``.
        norm_placement: ``"pre"`` (Pre-LN, default) or ``"post"`` (Post-LN).
    """
    d_model: int        = 128
    n_heads: int        = 4
    block_size: int     = 128
    d_ff: int | None    = None
    dropout: float      = 0.1
    norm_type: str      = "layernorm"
    activation: str     = "gelu"
    norm_placement: str = "pre"
    pos_type:       str       = "learned"  # "learned", "rope", "alibi", "gqa", "mqa", "gqa_rope"
    n_kv_heads:     int | None = None        # for GQA/MQA; None = MHA
    window_size:    int | None = None        # None = full attention; int = SWA window

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.dropout >= 0.0
        assert self.norm_type in ("layernorm", "rmsnorm")
        assert self.activation in ("gelu", "swiglu")
        assert self.norm_placement in ("pre", "post")
        assert self.pos_type in ("learned", "rope", "alibi")

    @property
    def effective_d_ff(self) -> int:
        """The FFN hidden dimension (resolves None to 4 * d_model)."""
        return self.d_ff or 4 * self.d_model
