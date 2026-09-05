"""
NanoMind — A GPT-style Language Model built layer by layer.

Quick start::

    from nanomind import NanoMind, ModelConfig

    cfg   = ModelConfig(vocab_size=256, d_model=128, n_layers=4, n_heads=4)
    model = NanoMind(cfg)
    logits, loss = model(idx, targets)

Version: 1.0.0
"""

__version__ = "2.3.0"
__author__  = "NanoMind Contributors"
__license__ = "MIT"

from nanomind.model import NanoMind, ModelConfig
from nanomind.config import NanoMindConfig
from nanomind.pos import get_attention, list_pos_types
from nanomind.lora import LoRAConfig, LoRAModel
from nanomind.speculative import SpeculativeConfig, SpeculativeGenerator
from nanomind.quant import QuantConfig, quantize_model
from nanomind.logging import LogConfig, TrainingLogger
from nanomind.generate.beam import BeamConfig, beam_search, diverse_beam_search
from nanomind.generate.beam_generator import BeamSearchGenerator
from nanomind.moe import MoEConfig, NanoMindMoE, SparseMoELayer
from nanomind.data import DataConfig, DataPipeline, InMemoryTokenDataset
from nanomind.cache import KVCacheConfig, NanoMindCached, CachedGenerator, KVCacheManager
from nanomind.flash import FlashConfig, FlashAttention, NanoMindFlash
from nanomind.amp import AMPConfig, AMPTrainer, GradAccumulator, mixed_precision_context

__all__ = [
    "NanoMind",
    "ModelConfig",
    "NanoMindConfig",
    "get_attention",
    "list_pos_types",
    "LoRAConfig",
    "LoRAModel",
    "SpeculativeConfig",
    "SpeculativeGenerator",
    "QuantConfig",
    "quantize_model",
    "LogConfig",
    "TrainingLogger",
    "BeamConfig",
    "beam_search",
    "diverse_beam_search",
    "BeamSearchGenerator",
    "MoEConfig",
    "NanoMindMoE",
    "SparseMoELayer",
    "DataConfig",
    "DataPipeline",
    "InMemoryTokenDataset",
    "KVCacheConfig",
    "NanoMindCached",
    "CachedGenerator",
    "KVCacheManager",
    "FlashConfig",
    "FlashAttention",
    "NanoMindFlash",
    "AMPConfig",
    "AMPTrainer",
    "GradAccumulator",
    "mixed_precision_context",
    "__version__",
]
