"""
config.py - Hyperparameter dataclasses for NanoMind
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Architecture hyperparameters for the transformer LLM."""

    vocab_size: int = 256
    block_size: int = 128
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
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
        emb = self.vocab_size * self.d_model
        per_layer = 12 * self.d_model * self.d_model + 2 * self.d_model
        final = self.d_model + self.vocab_size * self.d_model
        return emb + self.n_layers * per_layer + final


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    data_path: str = "data.txt"
    out_dir: str = "checkpoints"
    batch_size: int = 32
    max_iters: int = 5000
    eval_interval: int = 200
    eval_iters: int = 50
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    grad_clip: float = 1.0
    betas: tuple = field(default_factory=lambda: (0.9, 0.95))
    weight_decay: float = 0.1
    device: str = "auto"
    seed: int = 42
    log_interval: int = 10

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
