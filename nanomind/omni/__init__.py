"""
nanomind.omni — Unified Omni-Modal Foundation Architecture (NanoMind-Omni).
DIAMOND JUBILEE MAJOR RELEASE (v6.0.0).
"""
from nanomind.omni.config import (
    ModalityType,
    DuplexConfig,
    OmniConfig,
)
from nanomind.omni.interleave import (
    MultimodalItem,
    MultimodalSequenceBuilder,
)
from nanomind.omni.fusion import (
    ModalityEmbedding,
    CrossModalFusionLayer,
)
from nanomind.omni.duplex import (
    DialogueState,
    DuplexDialogueManager,
)
from nanomind.omni.turn_taking import (
    VoiceActivityDetector,
    EndOfUtterancePredictor,
)
from nanomind.omni.generator import (
    OmniGenerator,
)
from nanomind.omni.model import (
    NanoMindOmni,
)
from nanomind.omni.loss import (
    OmniMultiTaskLoss,
)
from nanomind.omni.benchmark import (
    OmniBenchmark,
)
from nanomind.omni.pipeline import (
    OmniResponse,
    OmniPipeline,
)

__all__ = [
    "ModalityType",
    "DuplexConfig",
    "OmniConfig",
    "MultimodalItem",
    "MultimodalSequenceBuilder",
    "ModalityEmbedding",
    "CrossModalFusionLayer",
    "DialogueState",
    "DuplexDialogueManager",
    "VoiceActivityDetector",
    "EndOfUtterancePredictor",
    "OmniGenerator",
    "NanoMindOmni",
    "OmniMultiTaskLoss",
    "OmniBenchmark",
    "OmniResponse",
    "OmniPipeline",
]
