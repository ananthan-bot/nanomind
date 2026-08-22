"""
nanomind/config.py — Top-level NanoMind configuration.

Combines ModelConfig, TrainConfig, CheckpointConfig, and GenerationConfig
into a single unified configuration that can be serialized to JSON/YAML.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

from nanomind.model.config import ModelConfig
from nanomind.trainer.config import TrainConfig
from nanomind.checkpoint.config import CheckpointConfig
from nanomind.generate.config import GenerationConfig


@dataclass
class NanoMindConfig:
    """
    Unified NanoMind configuration.

    Wraps all sub-configs so the entire experiment can be defined
    from a single JSON or YAML file and tracked as one artifact.

    Attributes:
        model:      Model architecture config.
        train:      Training hyperparameter config.
        checkpoint: Checkpoint management config.
        generate:   Text generation config.
        run_name:   Experiment name (used in checkpoint dir).
    """

    model:      ModelConfig      = field(default_factory=ModelConfig)
    train:      TrainConfig      = field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    generate:   GenerationConfig = field(default_factory=GenerationConfig)
    run_name:   str              = "nanomind_run"

    def to_dict(self) -> dict:
        """Serialize all sub-configs to a plain nested dict."""
        return {
            "model":      asdict(self.model),
            "train":      asdict(self.train),
            "checkpoint": asdict(self.checkpoint),
            "generate":   asdict(self.generate),
            "run_name":   self.run_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NanoMindConfig":
        """Deserialize from a nested dict."""
        return cls(
            model      = ModelConfig(**data.get("model", {})),
            train      = TrainConfig(**data.get("train", {})),
            checkpoint = CheckpointConfig(**data.get("checkpoint", {})),
            generate   = GenerationConfig(**data.get("generate", {})),
            run_name   = data.get("run_name", "nanomind_run"),
        )

    def save_json(self, path: str | Path) -> None:
        """Save unified config to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "NanoMindConfig":
        """Load unified config from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"NanoMindConfig("
            f"run='{self.run_name}', "
            f"d_model={self.model.d_model}, "
            f"n_layers={self.model.n_layers}, "
            f"max_iters={self.train.max_iters})"
        )
