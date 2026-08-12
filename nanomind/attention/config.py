"""
nanomind/attention/config.py — Configuration for the attention mechanism.
"""

from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """
    Configuration for :class:`~nanomind.attention.CausalSelfAttention`.

    Attributes:
        d_model:    Model embedding dimension.
        n_heads:    Number of attention heads.
        block_size: Maximum sequence length.
        dropout:    Attention and residual dropout probability.
        bias:       Whether to add bias to Q/K/V and output projections.
        use_flash:  Whether to use PyTorch 2.0 SDPA (Flash Attention) when available.
    """
    d_model: int    = 128
    n_heads: int    = 4
    block_size: int = 128
    dropout: float  = 0.1
    bias: bool      = False
    use_flash: bool = True

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.n_heads
