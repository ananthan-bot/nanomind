"""
nanomind/attention/swa.py — Sliding Window Attention (SWA).

Standard full attention has O(T²) memory and compute with sequence length T.
Sliding Window Attention limits each token to attending only to the W
previous tokens, reducing complexity to O(T × W):

  ┌──────────────────────────────────────────────────────┐
  │  Full attention       :  O(T²)   memory & compute    │
  │  Sliding Window (SWA) :  O(T·W)  memory & compute    │
  └──────────────────────────────────────────────────────┘

Each token can attend to up to W tokens in its local causal window.
Tokens more than W steps away are masked out (set to -∞ before softmax).

With multiple layers:
  - Layer 1: each token "sees" W tokens
  - Layer 2: each token effectively sees W² tokens (via layer 1 outputs)
  - Layer L: receptive field grows as W^L → full context with enough layers

Used in:
  - Mistral 7B (window_size=4096 on 8192 block_size)
  - Longformer (local + global attention)

Reference: Jiang et al. (2023) — https://arxiv.org/abs/2310.06825
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
