"""
config.py - Hyperparameter dataclasses for NanoMind
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Architecture hyperparameters for the transformer LLM."""
    vocab_size: int = 256
    block_size: int = 128
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    data_path: str = "data.txt"
    out_dir: str = "checkpoints"
    batch_size: int = 32
    max_iters: int = 5000
    learning_rate: float = 3e-4
    device: str = "auto"
    seed: int = 42
