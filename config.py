"""
config.py — Hyperparameter dataclasses for MiniGPT
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Architecture hyperparameters for the transformer LLM."""

    # Vocabulary size — set automatically by the tokenizer
    vocab_size: int = 256

    # Context window (sequence length)
    block_size: int = 128

    # Model width
    d_model: int = 128

    # Number of transformer layers
    n_layers: int = 4

    # Number of attention heads (d_model must be divisible by n_heads)
    n_heads: int = 4

    # Dropout probability (0.0 = no dropout)
    dropout: float = 0.1

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_params(self) -> int:
        """Rough parameter count (excluding positional embeddings)."""
        # Embeddings
        emb = self.vocab_size * self.d_model
        # Per layer: attn (4 * d_model^2) + FFN (8 * d_model^2) + 2 layernorms
        per_layer = 12 * self.d_model * self.d_model + 2 * self.d_model
        # Final layer norm + LM head
        final = self.d_model + self.vocab_size * self.d_model
        return emb + self.n_layers * per_layer + final


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Path to training text file
    data_path: str = "data.txt"

    # Checkpoint directory
    out_dir: str = "checkpoints"

    # Batch size (sequences per step)
    batch_size: int = 32

    # Total training iterations
    max_iters: int = 5000

    # Evaluation interval
    eval_interval: int = 200

    # Number of eval batches
    eval_iters: int = 50

    # Peak learning rate
    learning_rate: float = 3e-4

    # Minimum LR (cosine decay floor)
    min_lr: float = 3e-5

    # Warmup steps
    warmup_iters: int = 100

    # Gradient clipping max norm
    grad_clip: float = 1.0

    # AdamW betas
    betas: tuple = field(default_factory=lambda: (0.9, 0.95))

    # Weight decay
    weight_decay: float = 0.1

    # Device — 'auto', 'cpu', 'cuda', 'mps'
    device: str = "auto"

    # Random seed
    seed: int = 42

    # Log every N iters
    log_interval: int = 10

    # Save checkpoint every N evals
    save_interval: int = 1

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
