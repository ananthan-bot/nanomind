"""
tests/test_edit.py — Comprehensive unit tests for Model Editing algorithms (ROME / MEMIT).
"""
import math
import torch
import torch.nn as nn
from nanomind.edit import (
    KnowledgeFact,
    EditConfig,
    CausalTracer,
    KeyValueExtractor,
    CovarianceCollector,
    ROMEEngine,
    MEMITEngine,
    ModelEditor,
    EditEvaluator,
)


class TestKnowledgeFact:
    def test_fact_creation(self):
        fact = KnowledgeFact(
            prompt="The president of USA is",
            subject="USA",
            target_new="Alice",
            target_true="Bob",
            paraphrases=["The current leader of USA is"],
        )
        assert fact.subject == "USA"
        assert fact.target_new == "Alice"
        assert len(fact.paraphrases) == 1
