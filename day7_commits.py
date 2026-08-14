"""
day7_commits.py — 20 atomic commits for Day 7: Full NanoMind Model.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["GITHUB_TOKEN"] = ""

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== DAY 7: Full NanoMind Model — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — model package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/model/__init__.py", '"""NanoMind model sub-package."""\n')
commit("feat: add nanomind/model/ package skeleton")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — ModelConfig with all fields
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/model/config.py", '''\
"""
nanomind/model/config.py — NanoMind model configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Full configuration for a NanoMind transformer LLM.

    Attributes:
        vocab_size:     Size of the token vocabulary.
        block_size:     Maximum context window length (sequence length).
        d_model:        Embedding dimension.
        n_layers:       Number of stacked transformer blocks.
        n_heads:        Number of attention heads per block.
        d_ff:           FFN hidden dimension. None defaults to 4 * d_model.
        dropout:        Dropout probability (set to 0 for inference).
        norm_type:      Normalization — ``"layernorm"`` or ``"rmsnorm"``.
        activation:     FFN activation — ``"gelu"`` or ``"swiglu"``.
        norm_placement: ``"pre"`` (Pre-LN) or ``"post"`` (Post-LN).
        bias:           Whether to use bias in attention projections.
        weight_tying:   Tie token embedding weights to LM head.
    """

    vocab_size:     int       = 256
    block_size:     int       = 128
    d_model:        int       = 128
    n_layers:       int       = 4
    n_heads:        int       = 4
    d_ff:           int | None = None
    dropout:        float     = 0.1
    norm_type:      str       = "layernorm"
    activation:     str       = "gelu"
    norm_placement: str       = "pre"
    bias:           bool      = False
    weight_tying:   bool      = True
''')
commit("feat: add ModelConfig dataclass with all architecture fields")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — ModelConfig validation + properties
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/config.py")
src += '''
    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.n_layers > 0,   "n_layers must be positive"
        assert self.block_size > 0, "block_size must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.norm_type in ("layernorm", "rmsnorm")
        assert self.activation in ("gelu", "swiglu")
        assert self.norm_placement in ("pre", "post")

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.n_heads

    @property
    def effective_d_ff(self) -> int:
        """Resolved FFN hidden dimension."""
        return self.d_ff or 4 * self.d_model
'''
write("nanomind/model/config.py", src)
commit("feat: add ModelConfig validation in __post_init__ and head_dim, effective_d_ff properties")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — ModelConfig.from_dict / to_dict
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/config.py")
src += '''
    def to_dict(self) -> dict:
        """Serialize config to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Deserialize config from a plain dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
'''
write("nanomind/model/config.py", src)
commit("feat: add ModelConfig.to_dict() and from_dict() for serialization")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — ModelConfig.from_json / save_json
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/config.py")
src += '''
    def save_json(self, path: str | Path) -> None:
        """Save config to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        """Load config from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
'''
write("nanomind/model/config.py", src)
commit("feat: add ModelConfig.save_json() and from_json() for file persistence")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — NanoMind class skeleton + embeddings
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/model/model.py", '''\
"""
nanomind/model/model.py — The NanoMind GPT-style language model.

Architecture overview:
    Input IDs  (B, T)
        |
    Token Embedding   (vocab_size -> d_model)
    + Positional Emb  (block_size -> d_model)
        |
    [TransformerBlock] x N
        |
    Final LayerNorm
        |
    LM Head  (d_model -> vocab_size)   [weights tied to Token Embedding]
        |
    Logits  (B, T, vocab_size)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.model.config import ModelConfig
from nanomind.blocks import TransformerBlock, get_norm


class NanoMind(nn.Module):
    """
    NanoMind — A GPT-style causal language model.

    Args:
        cfg: :class:`~nanomind.model.ModelConfig` instance.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Token + position embeddings
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.block_size, cfg.d_model)
        self.emb_drop  = nn.Dropout(cfg.dropout)

        # Transformer blocks — to be filled in next commits
        self.blocks: nn.ModuleList | None = None

        # Final norm + LM head — to be filled in next commits
        self.final_norm: nn.Module | None = None
        self.lm_head:    nn.Linear | None = None
''')
commit("feat: add NanoMind class skeleton with token and positional embeddings")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — stack N transformer blocks
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/model.py")
src = src.replace(
    "        # Transformer blocks — to be filled in next commits\n"
    "        self.blocks: nn.ModuleList | None = None",
    """\
        # Stack of N transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                block_size=cfg.block_size,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                norm_type=cfg.norm_type,
                activation=cfg.activation,
                norm_placement=cfg.norm_placement,
            )
            for _ in range(cfg.n_layers)
        ])"""
)
write("nanomind/model/model.py", src)
commit("feat: stack N TransformerBlocks in NanoMind.__init__()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — final norm + LM head
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/model.py")
src = src.replace(
    "        # Final norm + LM head — to be filled in next commits\n"
    "        self.final_norm: nn.Module | None = None\n"
    "        self.lm_head:    nn.Linear | None = None",
    """\
        # Final normalization before the language model head
        self.final_norm = get_norm(cfg.norm_type, cfg.d_model)

        # LM head: projects d_model -> vocab_size
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)"""
)
write("nanomind/model/model.py", src)
commit("feat: add final LayerNorm and LM head projection to NanoMind")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — weight tying
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/model.py")
src = src.replace(
    "        # LM head: projects d_model -> vocab_size\n"
    "        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)",
    """\
        # LM head: projects d_model -> vocab_size
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share token embedding matrix with LM head
        # This reduces parameters and often improves performance.
        if cfg.weight_tying:
            self.lm_head.weight = self.token_emb.weight"""
)
write("nanomind/model/model.py", src)
commit("feat: add weight tying — LM head shares weights with token embedding")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — weight initialization (GPT-2 style)
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/model.py")
src = src.replace(
    "        # Weight tying: share token embedding matrix with LM head",
    """\
        # Initialize weights using GPT-2 style normal initialization
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layers) for stability
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

        # Weight tying: share token embedding matrix with LM head"""
)
src += '''
    # ── Initialization ────────────────────────────────────────────────────────

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 style weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
'''
write("nanomind/model/model.py", src)
commit("feat: add GPT-2 style weight initialization and residual projection scaling")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — forward() pass
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/model.py")
src += '''
    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Run a forward pass through NanoMind.

        Args:
            idx:     Input token IDs ``(B, T)``
            targets: Target token IDs ``(B, T)`` for loss computation.
                     If None, only logits are returned.

        Returns:
            Tuple of:
            - ``logits``: ``(B, T, vocab_size)``
            - ``loss``:   Cross-entropy loss scalar, or None if no targets.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"Sequence length {T} exceeds block_size {self.cfg.block_size}"
        )

        # Token + positional embeddings
        tok  = self.token_emb(idx)                                    # (B, T, d_model)
        pos  = self.pos_emb(torch.arange(T, device=idx.device))      # (T, d_model)
        x    = self.emb_drop(tok + pos)

        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x)

        # Final norm + LM head
        x      = self.final_norm(x)
        logits = self.lm_head(x)                                      # (B, T, vocab_size)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
            )
        return logits, loss
'''
write("nanomind/model/model.py", src)
commit("feat: implement NanoMind.forward() with embeddings, blocks, norm, and cross-entropy loss")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — num_parameters() utility
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/model.py")
src += '''
    # ── Utilities ─────────────────────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.

        Args:
            trainable_only: If True, count only trainable parameters
                            (requires_grad=True). Default: True.

        Returns:
            Total parameter count as an integer.
        """
        params = (
            p for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )
        return sum(p.numel() for p in params)
'''
write("nanomind/model/model.py", src)
commit("feat: add num_parameters() utility to NanoMind")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13 — __repr__ model summary
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/model.py")
src += '''
    def __repr__(self) -> str:
        from nanomind.utils.format import fmt_number
        n = self.num_parameters()
        return (
            f"NanoMind("
            f"vocab={self.cfg.vocab_size}, "
            f"d_model={self.cfg.d_model}, "
            f"layers={self.cfg.n_layers}, "
            f"heads={self.cfg.n_heads}, "
            f"params={fmt_number(n)})"
        )
'''
write("nanomind/model/model.py", src)
commit("feat: add NanoMind.__repr__() with compact model summary")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14 — autoregressive generate()
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/model/model.py")
src += '''
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """
        Autoregressively generate tokens appended to ``idx``.

        Args:
            idx:            Seed token IDs ``(B, T)``
            max_new_tokens: Number of new tokens to generate.
            temperature:    Softmax temperature. < 1 = sharper, > 1 = more random.
            top_k:          If set, only sample from the top-k logits.
            top_p:          If set, apply nucleus (top-p) sampling.

        Returns:
            Token IDs ``(B, T + max_new_tokens)``
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)
        return idx
'''
write("nanomind/model/model.py", src)
commit("feat: add autoregressive generate() with temperature, top-k, and top-p sampling")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 15 — update model __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/model/__init__.py", '''\
"""NanoMind model sub-package.

Primary exports:
    - :class:`NanoMind`    — the full GPT-style LLM
    - :class:`ModelConfig` — architecture configuration dataclass
"""

from nanomind.model.config import ModelConfig
from nanomind.model.model import NanoMind

__all__ = ["NanoMind", "ModelConfig"]
''')
commit("refactor: export NanoMind and ModelConfig from nanomind/model/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 16 — test: forward pass shapes
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_model.py", '''\
"""
tests/test_model.py — Tests for the NanoMind model.
"""

import pytest
import torch

from nanomind.model import NanoMind, ModelConfig

# Tiny config for fast tests
CFG = ModelConfig(
    vocab_size=64,
    block_size=16,
    d_model=32,
    n_layers=2,
    n_heads=2,
    dropout=0.0,
)
B, T = 2, 8


@pytest.fixture
def model() -> NanoMind:
    torch.manual_seed(0)
    return NanoMind(CFG)


# ── Forward pass shapes ───────────────────────────────────────────────────────

class TestForwardShape:
    def test_logits_shape(self, model):
        idx = torch.randint(0, CFG.vocab_size, (B, T))
        logits, loss = model(idx)
        assert logits.shape == (B, T, CFG.vocab_size)

    def test_no_targets_loss_is_none(self, model):
        idx = torch.randint(0, CFG.vocab_size, (B, T))
        _, loss = model(idx)
        assert loss is None

    def test_with_targets_loss_is_scalar(self, model):
        idx     = torch.randint(0, CFG.vocab_size, (B, T))
        targets = torch.randint(0, CFG.vocab_size, (B, T))
        _, loss = model(idx, targets)
        assert loss is not None
        assert loss.shape == ()   # scalar

    def test_single_token(self, model):
        idx = torch.randint(0, CFG.vocab_size, (1, 1))
        logits, _ = model(idx)
        assert logits.shape == (1, 1, CFG.vocab_size)

    def test_full_block_size(self, model):
        idx = torch.randint(0, CFG.vocab_size, (B, CFG.block_size))
        logits, _ = model(idx)
        assert logits.shape == (B, CFG.block_size, CFG.vocab_size)
''')
commit("test: add NanoMind forward pass shape tests (logits, loss, single token)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 17 — test: loss computation
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_model.py")
src += '''

# ── Loss computation ──────────────────────────────────────────────────────────

class TestLossComputation:
    def test_loss_is_positive(self, model):
        idx     = torch.randint(0, CFG.vocab_size, (B, T))
        targets = torch.randint(0, CFG.vocab_size, (B, T))
        _, loss = model(idx, targets)
        assert loss.item() > 0

    def test_loss_decreases_with_correct_targets(self, model):
        # Force a simple case: one token, targets = input (should be easy)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        idx     = torch.tensor([[5, 10, 15]])
        targets = torch.tensor([[10, 15, 5]])
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            _, loss = model(idx, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # Loss should go down over 30 steps
        assert losses[-1] < losses[0]

    def test_random_loss_near_log_vocab(self, model):
        # Untrained model: loss should be ~log(vocab_size) = log(64) ≈ 4.15
        import math
        idx     = torch.randint(0, CFG.vocab_size, (4, T))
        targets = torch.randint(0, CFG.vocab_size, (4, T))
        _, loss = model(idx, targets)
        expected = math.log(CFG.vocab_size)
        assert abs(loss.item() - expected) < 1.0
'''
write("tests/test_model.py", src)
commit("test: add loss computation tests (positive, decreases, near log-uniform)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 18 — test: weight tying
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_model.py")
src += '''

# ── Weight tying ──────────────────────────────────────────────────────────────

class TestWeightTying:
    def test_weights_are_same_object(self, model):
        # With weight_tying=True, lm_head.weight IS token_emb.weight
        assert model.lm_head.weight is model.token_emb.weight

    def test_no_weight_tying(self):
        cfg = ModelConfig(
            vocab_size=64, block_size=16, d_model=32,
            n_layers=2, n_heads=2, dropout=0.0, weight_tying=False
        )
        m = NanoMind(cfg)
        assert m.lm_head.weight is not m.token_emb.weight

    def test_tying_reduces_params(self):
        tied   = NanoMind(ModelConfig(vocab_size=64, block_size=16, d_model=32,
                                      n_layers=2, n_heads=2, dropout=0.0, weight_tying=True))
        untied = NanoMind(ModelConfig(vocab_size=64, block_size=16, d_model=32,
                                      n_layers=2, n_heads=2, dropout=0.0, weight_tying=False))
        assert tied.num_parameters() < untied.num_parameters()
'''
write("tests/test_model.py", src)
commit("test: add weight tying tests (same object, untied variant, parameter count)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — test: generate() + ModelConfig serialization
# ══════════════════════════════════════════════════════════════════════════════
src = read("tests/test_model.py")
src += '''

# ── Generation ────────────────────────────────────────────────────────────────

class TestGenerate:
    def test_generate_length(self, model):
        idx = torch.randint(0, CFG.vocab_size, (1, 4))
        out = model.generate(idx, max_new_tokens=5)
        assert out.shape == (1, 9)   # 4 seed + 5 new

    def test_generate_is_deterministic_with_temperature_zero(self, model):
        idx  = torch.randint(0, CFG.vocab_size, (1, 4))
        out1 = model.generate(idx, max_new_tokens=5, temperature=1e-8, top_k=1)
        out2 = model.generate(idx, max_new_tokens=5, temperature=1e-8, top_k=1)
        assert torch.equal(out1, out2)

    def test_generate_stays_in_vocab(self, model):
        idx = torch.randint(0, CFG.vocab_size, (1, 4))
        out = model.generate(idx, max_new_tokens=10)
        assert out.max().item() < CFG.vocab_size
        assert out.min().item() >= 0


# ── ModelConfig serialization ─────────────────────────────────────────────────

class TestModelConfig:
    def test_to_from_dict_roundtrip(self):
        cfg  = ModelConfig(vocab_size=100, d_model=64, n_layers=3)
        cfg2 = ModelConfig.from_dict(cfg.to_dict())
        assert cfg == cfg2

    def test_save_load_json(self, tmp_path):
        cfg  = ModelConfig(vocab_size=100, d_model=64, n_layers=3)
        p    = tmp_path / "cfg.json"
        cfg.save_json(p)
        cfg2 = ModelConfig.from_json(p)
        assert cfg == cfg2

    def test_invalid_config_raises(self):
        with pytest.raises(AssertionError):
            ModelConfig(d_model=65, n_heads=4)   # not divisible
'''
write("tests/test_model.py", src)
commit("test: add generate() correctness tests and ModelConfig serialization roundtrip")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — update README roadmap + CHANGELOG
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| 7 | Full model | 🔜 |",
    "| 7 | Full model | ✅ Done — 20 commits |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = cl.replace(
    "- Blocks: TransformerBlock (Pre/Post-LN), FeedForward (GELU/SwiGLU), RMSNorm, LayerNorm (Day 6)",
    "- Blocks: TransformerBlock (Pre/Post-LN), FeedForward (GELU/SwiGLU), RMSNorm, LayerNorm (Day 6)\n- Full model: NanoMind with embeddings, N blocks, weight tying, generate(), ModelConfig (Day 7)"
)
write("CHANGELOG.md", cl)
commit("chore: mark Day 7 complete in README and CHANGELOG — Week 1 architecture done!")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 7 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")
print("=== DAY 7 COMPLETE — WEEK 1 DONE! ===")
