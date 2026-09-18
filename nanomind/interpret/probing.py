"""
nanomind/interpret/probing.py — Linear probing classifiers.

## What is Probing?

Probing tests whether a specific linguistic concept (e.g., POS tags,
syntactic depth, coreference) is linearly decodable from internal
representations.

Protocol:
  1. Freeze the language model
  2. Extract hidden states at layer L for each token/sentence
  3. Train a linear classifier on top of these representations
  4. Measure classification accuracy

High accuracy → the concept is encoded in layer L.
Low accuracy  → the concept is not (linearly) accessible at layer L.

Used to understand:
  - Which layers encode syntax vs semantics
  - How representations evolve through depth
  - What information is lost in compression

References:
  Alain & Bengio (2016) "Understanding Intermediate Layers Using Linear Classifier Probes"
  Tenney et al. (2019) "BERT Rediscovers the Classical NLP Pipeline"
  https://arxiv.org/abs/1905.05950
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ProbeResult:
    """Result of a linear probe experiment."""
    layer:    int
    task:     str
    accuracy: float
    loss:     float
    n_train:  int
    n_test:   int

    def to_dict(self) -> dict:
        return {
            "layer":    self.layer,
            "task":     self.task,
            "accuracy": round(self.accuracy, 4),
            "loss":     round(self.loss, 4),
            "n_train":  self.n_train,
            "n_test":   self.n_test,
        }


class LinearProbe(nn.Module):
    """
    Linear probing classifier on top of frozen representations.

    Args:
        d_model:   Input feature dimension.
        n_classes: Number of probe classes.

    Example::

        probe  = LinearProbe(d_model=256, n_classes=10)
        result = probe.fit(train_X, train_y, test_X, test_y)
    """

    def __init__(self, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def fit(
        self,
        train_X: torch.Tensor,
        train_y: torch.Tensor,
        test_X:  torch.Tensor,
        test_y:  torch.Tensor,
        lr:      float = 1e-2,
        epochs:  int   = 30,
        layer:   int   = 0,
        task:    str   = "probe",
    ) -> ProbeResult:
        """
        Train the probe and evaluate on test set.

        Args:
            train_X: ``(N_train, D)`` representations.
            train_y: ``(N_train,)`` integer labels.
            test_X:  ``(N_test, D)`` representations.
            test_y:  ``(N_test,)`` integer labels.

        Returns:
            :class:`ProbeResult`.
        """
        opt     = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        self.train()
        final_loss = 0.0
        for _ in range(epochs):
            opt.zero_grad()
            logits = self(train_X)
            loss   = F.cross_entropy(logits, train_y)
            loss.backward()
            opt.step()
            final_loss = loss.item()

        self.eval()
        with torch.no_grad():
            preds    = self(test_X).argmax(dim=-1)
            accuracy = (preds == test_y).float().mean().item()

        return ProbeResult(
            layer    = layer,
            task     = task,
            accuracy = accuracy,
            loss     = final_loss,
            n_train  = len(train_X),
            n_test   = len(test_X),
        )


class LayerwiseProber:
    """
    Run linear probes across all layers of a model.

    Extracts hidden states at each layer and trains a probe per layer,
    revealing how probe accuracy changes with depth.

    Args:
        model:     Language model.
        n_classes: Number of probe classes.
        task:      Task name (for logging).

    Example::

        prober  = LayerwiseProber(model, n_classes=5, task="pos_tags")
        results = prober.probe_all_layers(X_ids, labels)
        # results[i].accuracy → probe accuracy at layer i
    """

    def __init__(self, model: nn.Module, n_classes: int, task: str = "probe") -> None:
        self.model     = model
        self.n_classes = n_classes
        self.task      = task

    @torch.no_grad()
    def extract_hiddens(
        self, input_ids: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Extract hidden states at each transformer block.

        Returns:
            List of ``(T, D)`` tensors, one per layer.
        """
        hiddens = []
        x       = input_ids

        # Extract via hooks
        def hook_fn(m, inp, out):
            if isinstance(out, torch.Tensor):
                hiddens.append(out.detach().cpu())

        hooks   = []
        for m in self.model.modules():
            if isinstance(m, nn.LayerNorm):
                hooks.append(m.register_forward_hook(hook_fn))

        self.model(x)
        for h in hooks:
            h.remove()

        return hiddens

    def probe_all_layers(
        self,
        input_ids: torch.Tensor,
        labels:    torch.Tensor,
        train_frac: float = 0.8,
        epochs:    int    = 20,
    ) -> list[ProbeResult]:
        """
        Probe each layer with a linear classifier.

        Args:
            input_ids: ``(N, T)`` token ID batch.
            labels:    ``(N,)`` integer labels per sample.
            train_frac: Train split fraction.

        Returns:
            List of :class:`ProbeResult`, one per layer.
        """
        results  = []
        n_train  = max(1, int(len(input_ids) * train_frac))
        train_X_ids, test_X_ids = input_ids[:n_train], input_ids[n_train:]
        train_y, test_y         = labels[:n_train], labels[n_train:]

        if len(test_X_ids) == 0:
            test_X_ids = train_X_ids
            test_y     = train_y

        train_hiddens = self.extract_hiddens(train_X_ids)
        test_hiddens  = self.extract_hiddens(test_X_ids)

        n_layers = min(len(train_hiddens), len(test_hiddens))
        for layer_idx in range(n_layers):
            th = train_hiddens[layer_idx]
            eh = test_hiddens[layer_idx]
            # Mean-pool over sequence
            if th.dim() == 3:
                th = th.mean(dim=1)   # (N, D)
                eh = eh.mean(dim=1)
            elif th.dim() == 2:
                continue  # skip scalar layers

            d_model = th.shape[-1]
            probe   = LinearProbe(d_model, self.n_classes)
            result  = probe.fit(
                th, train_y[:len(th)],
                eh, test_y[:len(eh)],
                layer=layer_idx, task=self.task, epochs=epochs,
            )
            results.append(result)

        return results
