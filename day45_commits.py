"""
day45_commits.py — 20 atomic commits for Day 45: Long-Context & Efficient Attention.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

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

print("\n=== DAY 45: Long-Context & Efficient Attention — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — longctx package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/longctx/__init__.py",
      '"""NanoMind Long-Context sub-package — Efficient Attention for long sequences."""\n')
commit("feat: add nanomind/longctx/ package skeleton for long-context efficient attention")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — RoPE (Rotary Position Embeddings)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/longctx/rope.py", '''\
"""
nanomind/longctx/rope.py — Rotary Position Embeddings (RoPE).

## RoPE: Rotary Position Embedding (Su et al., 2021)

Standard position embeddings add position info to token embeddings:
  h = token_emb + pos_emb   (absolute, fixed max length)

RoPE instead rotates query and key vectors based on their position:
  q_m = R(m) q   k_n = R(n) k
  <q_m, k_n> depends only on (q, k, m-n) — relative position!

The rotation matrix R(m) for a 2D subspace:
  [cos(m θ)  -sin(m θ)]
  [sin(m θ)   cos(m θ)]

For D-dimensional vectors, applied in pairs:
  (q_{2i}, q_{2i+1}) → rotated by m × θ_i
  where θ_i = base^{-2i/D}  (default base=10000)

Benefits over learned positional embeddings:
  ✓ Relative position awareness (m-n)
  ✓ No max-length limit at training time
  ✓ Generalises beyond training length (with scaling)
  ✓ Used by: LLaMA, Mistral, GPT-NeoX, PaLM, Gemma, Qwen

## RoPE Scaling for Long Contexts

Standard RoPE fails beyond training length due to OOD frequencies.
Extension techniques:
  - Linear scaling:  θ_i → θ_i / scale_factor  (Code LLaMA)
  - NTK-aware:       base → base × scale^{D/(D-2)}  (LLaMA.cpp)
  - YaRN:            frequency-domain scaling (Mistral 32K)
  - LongRoPE:        non-uniform scaling per frequency

References:
  Su et al. (2021) "RoFormer" https://arxiv.org/abs/2104.09864
  Chen et al. (2023) "Extending Context Window" https://arxiv.org/abs/2306.15595
"""

from __future__ import annotations
import math
import torch


class RotaryEmbedding:
    """
    Rotary Position Embeddings (RoPE).

    Pre-computes cos/sin tables for fast application.

    Args:
        dim:          Head dimension (must be even).
        base:         Frequency base (default 10000, LLaMA uses 500000).
        max_seq:      Pre-compute up to this length.
        scale_factor: Linear scaling for context extension (1.0 = standard).
        scaling_type: ``"none"``, ``"linear"``, ``"ntk"``, or ``"yarn"``.

    Example::

        rope = RotaryEmbedding(dim=64, max_seq=4096)
        q, k = rope.apply(q, k, seq_len=512)
    """

    def __init__(
        self,
        dim:          int,
        base:         float = 10000.0,
        max_seq:      int   = 4096,
        scale_factor: float = 1.0,
        scaling_type: str   = "none",
    ) -> None:
        assert dim % 2 == 0, "RoPE dim must be even"
        self.dim          = dim
        self.base         = base
        self.max_seq      = max_seq
        self.scale_factor = scale_factor
        self.scaling_type = scaling_type
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._build_cache(max_seq)

    def _get_freqs(self) -> torch.Tensor:
        """Compute inverse frequencies with optional scaling."""
        half = self.dim // 2
        if self.scaling_type == "ntk":
            # NTK-aware scaling: adjust base
            base = self.base * (self.scale_factor ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        else:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            if self.scaling_type == "linear" and self.scale_factor != 1.0:
                inv_freq = inv_freq / self.scale_factor
        return inv_freq

    def _build_cache(self, max_seq: int) -> None:
        """Pre-compute cos/sin tables."""
        inv_freq = self._get_freqs()                         # (D/2,)
        t        = torch.arange(max_seq).float()             # (T,)
        freqs    = torch.outer(t, inv_freq)                  # (T, D/2)
        emb      = torch.cat([freqs, freqs], dim=-1)         # (T, D)
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the last dimension: [x1, x2] → [-x2, x1]."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply(
        self,
        q:       torch.Tensor,
        k:       torch.Tensor,
        seq_len: int,
        offset:  int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to queries and keys.

        Args:
            q:       ``(B, H, T, D)`` query tensor.
            k:       ``(B, H, T, D)`` key tensor.
            seq_len: Current sequence length.
            offset:  Position offset (for KV cache).

        Returns:
            ``(q_rot, k_rot)`` with rotary embeddings applied.
        """
        if seq_len + offset > self.max_seq:
            self._build_cache(seq_len + offset + 1)
            self.max_seq = seq_len + offset + 1

        cos = self._cos_cached[offset: offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cached[offset: offset + seq_len].unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot

    def extend(self, new_max: int) -> None:
        """Extend the pre-computed cache to a new maximum length."""
        if new_max > self.max_seq:
            self._build_cache(new_max)
            self.max_seq = new_max

    def to_dict(self) -> dict:
        return {
            "dim":          self.dim,
            "base":         self.base,
            "max_seq":      self.max_seq,
            "scale_factor": self.scale_factor,
            "scaling_type": self.scaling_type,
        }
''')
commit("feat: add RotaryEmbedding (RoPE) — linear/NTK scaling, apply(q,k), pre-computed cos/sin cache")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — ALiBi (Attention with Linear Biases)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/longctx/alibi.py", '''\
"""
nanomind/longctx/alibi.py — ALiBi: Attention with Linear Biases.

## ALiBi (Press et al., 2021)

Instead of positional embeddings, ALiBi adds a linear bias
to attention scores based on the relative distance between tokens:

  Attention(Q, K, V) = softmax(QK^T / √d + m × (-|i - j|)) × V

where m is a head-specific slope:
  slopes = [2^{-8/n}, 2^{-16/n}, ..., 2^{-8}] for n heads

Key advantages:
  ✓ No position embeddings at all → simpler
  ✓ Strong length generalisation (trained on 1024, works on 4096+)
  ✓ Used by BLOOM (176B params), MPT, OPT-66B

Key limitation:
  - No relative position rotations (RoPE) → slightly worse on instruction tasks

Reference:
  Press et al. (2022) "Train Short, Test Long: Attention with Linear Biases"
  https://arxiv.org/abs/2108.12409
"""

from __future__ import annotations
import math
import torch


class ALiBi:
    """
    ALiBi position bias for attention.

    Pre-computes per-head slopes and bias matrices.

    Args:
        n_heads:  Number of attention heads.
        max_seq:  Pre-compute bias up to this length.

    Example::

        alibi   = ALiBi(n_heads=8, max_seq=2048)
        # Apply to attention scores before softmax:
        scores  = (q @ k.T) / math.sqrt(d_head)
        scores  = scores + alibi.bias(seq_len=512)
        weights = softmax(scores)
    """

    def __init__(self, n_heads: int, max_seq: int = 2048) -> None:
        self.n_heads = n_heads
        self.max_seq = max_seq
        self.slopes  = self._get_slopes(n_heads)    # (H,)
        self._bias_cache: dict[int, torch.Tensor] = {}

    @staticmethod
    def _get_slopes(n_heads: int) -> torch.Tensor:
        """Compute ALiBi slopes for each attention head."""
        def _slopes_power_of_2(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n_heads).is_integer():
            return torch.tensor(_slopes_power_of_2(n_heads))
        else:
            # Nearest power of 2
            n_pow2  = 2 ** math.floor(math.log2(n_heads))
            slopes  = _slopes_power_of_2(n_pow2)
            extra   = _slopes_power_of_2(2 * n_pow2)
            slopes  = slopes + extra[0::2][:n_heads - n_pow2]
            return torch.tensor(slopes[:n_heads])

    def bias(self, seq_len: int) -> torch.Tensor:
        """
        Compute ALiBi bias matrix for a given sequence length.

        Args:
            seq_len: Current sequence length.

        Returns:
            ``(H, T, T)`` bias tensor (negative values, causal mask compatible).
        """
        if seq_len in self._bias_cache:
            return self._bias_cache[seq_len]

        # Relative distances: bias[i,j] = -(i - j) for i >= j
        pos    = torch.arange(seq_len)
        dist   = pos.unsqueeze(0) - pos.unsqueeze(1)    # (T, T)
        # Only apply to past (causal): negative distances → -inf for future
        dist   = dist.abs().float()
        bias   = -dist.unsqueeze(0) * self.slopes.view(-1, 1, 1)  # (H, T, T)
        # Mask future positions
        causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        bias   = bias.masked_fill(causal.unsqueeze(0), float("-inf"))
        self._bias_cache[seq_len] = bias
        return bias

    def apply_to_scores(
        self,
        scores:  torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Add ALiBi bias to attention scores.

        Args:
            scores:  ``(B, H, T, T)`` raw attention scores.
            seq_len: Sequence length T.

        Returns:
            ``(B, H, T, T)`` biased scores.
        """
        bias = self.bias(seq_len)   # (H, T, T)
        return scores + bias.unsqueeze(0)

    def to_dict(self) -> dict:
        return {
            "n_heads":  self.n_heads,
            "max_seq":  self.max_seq,
            "slopes":   self.slopes.tolist(),
        }
''')
commit("feat: add ALiBi — per-head slopes, causal bias matrix, apply_to_scores(), to_dict()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — Sliding window attention
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/longctx/sliding_window.py", '''\
"""
nanomind/longctx/sliding_window.py — Sliding Window Attention.

## Sliding Window Attention (Longformer, Mistral)

Standard attention: O(T²) memory and compute.
For long sequences (T=32K), this becomes infeasible.

Sliding window attention restricts each token to attend only to
a local window of W neighbours:
  token i attends to tokens [i-W/2, ..., i+W/2]

This reduces complexity to O(T × W).

Key insight: information still propagates globally through stacking.
With L layers and window W, effective receptive field = L × W.

With window W=4096 and L=32 layers:
  Effective field = 32 × 4096 = 131,072 tokens

Used by: Mistral 7B (W=4096), Longformer (W=512), BigBird.

## Attention Sinks

(Xiao et al., 2023) observed that LLMs always attend strongly to
the very first tokens (attention sinks). This is because:
  - Initial tokens see all future tokens during training
  - Models use them as "resting state" or "memory dump"

StreamingLLM keeps the first K_sink tokens (sinks) + recent W tokens:
  Total window = K_sink + W
  This enables infinite-length generation with bounded memory!

References:
  Beltagy et al. (2020) Longformer: https://arxiv.org/abs/2004.05150
  Jiang et al. (2023) Mistral: https://arxiv.org/abs/2310.06825
  Xiao et al. (2023) StreamingLLM: https://arxiv.org/abs/2309.17453
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    """
    Multi-head attention with sliding window constraint.

    Each token only attends to the nearest W tokens,
    with optional attention sink tokens (always attended to).

    Args:
        d_model:     Model dimension.
        n_heads:     Number of attention heads.
        window_size: Local attention window W.
        n_sinks:     Number of sink tokens (always attended to).
        dropout:     Attention dropout.

    Example::

        swa = SlidingWindowAttention(d_model=512, n_heads=8, window_size=256)
        out = swa(x)   # x: (B, T, D)  — efficient O(T×W) attention
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        window_size: int,
        n_sinks:     int   = 4,
        dropout:     float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.d_head      = d_model // n_heads
        self.window_size = window_size
        self.n_sinks     = n_sinks

        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.o_proj  = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale   = self.d_head ** -0.5

    def _make_window_mask(self, T: int) -> torch.Tensor:
        """
        Create sliding window causal mask.

        mask[i, j] = True if token j is in window of token i (and j <= i).

        Returns:
            ``(T, T)`` boolean tensor.
        """
        # Causal mask
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        # Window mask: only attend to last W positions
        dist   = torch.arange(T).unsqueeze(0) - torch.arange(T).unsqueeze(1)
        in_win = dist >= -self.window_size
        # Sink tokens: always attended to
        sinks  = torch.zeros(T, T, dtype=torch.bool)
        if self.n_sinks > 0:
            sinks[:, :self.n_sinks] = True
        return causal & (in_win | sinks)

    def forward(
        self,
        x:            torch.Tensor,
        past_kv:      tuple | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Sliding window attention forward.

        Args:
            x:       ``(B, T, D)`` input tensor.
            past_kv: Optional cached ``(k, v)`` for generation.

        Returns:
            ``(output, (k, v))``
        """
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)   # (B,H,T,Dh)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        T_full = k.shape[2]
        scores = (q @ k.transpose(-2, -1)) * self.scale        # (B,H,T,T_full)

        # Apply sliding window + causal mask
        mask   = self._make_window_mask(T_full)                 # (T_full, T_full)
        # For generation (T < T_full), take last T rows
        if T < T_full:
            mask = mask[T_full - T:]
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out     = (weights @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out), (k, v)

    def effective_context(self, n_layers: int) -> int:
        """Effective receptive field across L layers."""
        return n_layers * self.window_size + self.n_sinks
''')
commit("feat: add SlidingWindowAttention — window mask, attention sinks, past_kv cache, effective_context()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — Linear attention (Performer)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/longctx/linear_attn.py", '''\
"""
nanomind/longctx/linear_attn.py — Linear Attention (O(T) complexity).

## Linear Attention

Standard attention: softmax(QK^T/√d)V requires O(T²) memory.

Linear attention approximates this with kernel functions:
  Attn(Q, K, V) = φ(Q) × (φ(K)^T V) / φ(Q) × (φ(K)^T 1)

where φ is a feature map (kernel trick).

This rewrites the computation order:
  Instead of: (φ(Q)(φ(K)^T)) V  [O(T²D)]
  Compute:    φ(Q)(φ(K)^T V)    [O(TD²)] — much faster for T >> D!

## Feature Maps

ELU+1 (Katharopoulos et al., 2020):
  φ(x) = ELU(x) + 1  (non-negative, simple)

Random Features / Performer (Choromanski et al., 2021):
  φ(x) = exp(ω^T x - ||x||²/2) × random_features
  Unbiased approximation to softmax kernel

RetNet (Sun et al., 2023):
  Combines linear attention with recurrent formulation.
  γ decay factor: each position decays by γ^distance.

## Causal Linear Attention

For causal (left-to-right) attention:
  h_t = φ(Q_t) × Σ_{s≤t} φ(K_s) ⊗ V_s / φ(Q_t) × Σ_{s≤t} φ(K_s)

This has an exact recurrent form:
  S_t = S_{t-1} + φ(K_t) ⊗ V_t    [state matrix, D×D]
  z_t = z_{t-1} + φ(K_t)            [normaliser, D]
  h_t = φ(Q_t) S_t / φ(Q_t) z_t

O(T) time and O(D²) memory — constant in sequence length!

References:
  Katharopoulos et al. (2020) https://arxiv.org/abs/2006.16236
  Choromanski et al. (2021) Performer: https://arxiv.org/abs/2009.14794
  Sun et al. (2023) RetNet: https://arxiv.org/abs/2307.08621
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU+1 feature map: φ(x) = ELU(x) + 1 (non-negative)."""
    return F.elu(x) + 1.0


def relu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ReLU feature map: φ(x) = ReLU(x) (non-negative, sparse)."""
    return F.relu(x)


class LinearAttention(nn.Module):
    """
    Linear Attention: O(T) complexity via kernel approximation.

    Uses the ELU+1 feature map for the kernel approximation.
    Supports both parallel (training) and recurrent (inference) modes.

    Args:
        d_model:  Model dimension.
        n_heads:  Number of attention heads.
        feature:  Feature map: ``"elu"`` or ``"relu"``.
        eps:      Normalisation epsilon.

    Example::

        la  = LinearAttention(d_model=256, n_heads=8)
        out = la(x)   # x: (B, T, D) — O(T) complexity!
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        feature: str   = "elu",
        eps:     float = 1e-6,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.eps     = eps
        self.feature = elu_feature_map if feature == "elu" else relu_feature_map

        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.o_proj  = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x:    torch.Tensor,
        mode: str = "parallel",
    ) -> torch.Tensor:
        """
        Linear attention forward pass.

        Args:
            x:    ``(B, T, D)`` input.
            mode: ``"parallel"`` (O(TD²)) or ``"recurrent"`` (O(TD²) but constant memory).

        Returns:
            ``(B, T, D)`` output.
        """
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        Q = self.feature(self.q_proj(x).view(B, T, H, Dh))   # (B,T,H,Dh)
        K = self.feature(self.k_proj(x).view(B, T, H, Dh))
        V = self.v_proj(x).view(B, T, H, Dh)

        if mode == "parallel":
            out = self._parallel(Q, K, V)
        else:
            out = self._recurrent(Q, K, V)

        out = out.view(B, T, D)
        return self.o_proj(out)

    def _parallel(self, Q, K, V):
        """Parallel causal linear attention."""
        B, T, H, Dh = Q.shape
        out = torch.zeros_like(Q)
        for h in range(H):
            Qh = Q[:, :, h, :]    # (B, T, Dh)
            Kh = K[:, :, h, :]
            Vh = V[:, :, h, :]
            S  = torch.zeros(B, Dh, Dh)    # state: K^T V
            z  = torch.zeros(B, Dh)        # normaliser
            step_out = torch.zeros(B, T, Dh)
            for t in range(T):
                S = S + torch.bmm(Kh[:, t:t+1].transpose(1, 2),
                                   Vh[:, t:t+1])            # (B, Dh, Dh)
                z = z + Kh[:, t, :]                          # (B, Dh)
                q_t = Qh[:, t:t+1, :]                       # (B, 1, Dh)
                num = torch.bmm(q_t, S).squeeze(1)          # (B, Dh)
                den = (q_t.squeeze(1) * z).sum(-1, keepdim=True) + self.eps
                step_out[:, t, :] = num / den
            out[:, :, h, :] = step_out
        return out

    def _recurrent(self, Q, K, V):
        """Recurrent causal linear attention (same output, sequential)."""
        return self._parallel(Q, K, V)   # same result, alias for clarity

    @property
    def complexity(self) -> str:
        return f"O(T × D²) = O(T × {self.d_head**2}) — linear in T"


class RetNetDecay(nn.Module):
    """
    RetNet-style decayed linear attention (Sun et al., 2023).

    Adds a per-head exponential decay γ^(i-j) to favour recent context.

    Args:
        d_model:    Model dimension.
        n_heads:    Number of heads.
        gamma_min:  Minimum decay (head 0).
        gamma_max:  Maximum decay (last head).
    """

    def __init__(
        self,
        d_model:   int,
        n_heads:   int,
        gamma_min: float = 0.8,
        gamma_max: float = 0.999,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        # Per-head decay gammas
        gammas = torch.linspace(gamma_min, gamma_max, n_heads)
        self.register_buffer = lambda n, t: setattr(self, n, t)
        self.gammas          = gammas

        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.o_proj  = nn.Linear(d_model, d_model, bias=False)

    def _decay_mask(self, T: int, gamma: float) -> torch.Tensor:
        """Causal decay mask: γ^(i-j) for j<=i, else 0."""
        pos  = torch.arange(T).unsqueeze(0) - torch.arange(T).unsqueeze(1)
        mask = torch.where(pos >= 0, gamma ** pos.float(), torch.zeros(T, T))
        return mask   # (T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # (B,H,T,Dh)
        K = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        # Scaled dot-product with decay
        scale = Dh ** -0.5
        out   = torch.zeros(B, H, T, Dh)
        for h in range(H):
            dm    = self._decay_mask(T, self.gammas[h].item())   # (T,T)
            sc    = (Q[:, h] @ K[:, h].transpose(-1, -2)) * scale * dm
            sc    = sc / (sc.abs().sum(-1, keepdim=True) + 1e-6)
            out[:, h] = sc @ V[:, h]

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)
''')
commit("feat: add LinearAttention (ELU/ReLU kernels, O(T) recurrent), RetNetDecay (per-head gamma)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Grouped Query Attention (GQA) and Multi-Query Attention (MQA)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/longctx/gqa.py", '''\
"""
nanomind/longctx/gqa.py — Grouped Query Attention (GQA) and Multi-Query Attention (MQA).

## The KV Cache Bottleneck

During autoregressive generation, each step adds one new (K, V) pair per head.
With H heads, sequence length T, and batch B:
  KV cache = 2 × B × H × T × D_head bytes

For LLaMA-65B: 2 × 40 heads × 4096 × 128 bytes × batch_size — enormous!

## Multi-Query Attention (MQA, Shazeer 2019)

Share a single K, V across all Q heads:
  - H query heads (each with its own Q projection)
  - 1 key head  (shared across all Q heads)
  - 1 value head (shared across all Q heads)

KV cache reduction: H× smaller!
Quality: slightly worse than MHA, but faster inference.

Used by: PaLM, Falcon, StarCoder.

## Grouped Query Attention (GQA, Ainslie et al., 2023)

Compromise between MHA and MQA:
  - H query heads split into G groups
  - Each group shares 1 K, V head
  - G KV heads total

KV cache reduction: H/G× smaller!
Quality: nearly identical to MHA.

Used by: LLaMA-2 70B, Mistral, Gemma, Command-R.

Reference:
  Shazeer (2019) MQA: https://arxiv.org/abs/1911.02150
  Ainslie et al. (2023) GQA: https://arxiv.org/abs/2305.13245
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    Generalises MHA (n_kv_heads=n_heads) and MQA (n_kv_heads=1).

    Args:
        d_model:     Model dimension.
        n_heads:     Number of query heads (H).
        n_kv_heads:  Number of KV heads (G). Must divide H evenly.
        max_seq:     Maximum sequence length.
        dropout:     Attention dropout.
        use_rope:    Apply RoPE position encoding.

    Example::

        # LLaMA-2 70B: 64 Q heads, 8 KV heads
        gqa = GroupedQueryAttention(d_model=8192, n_heads=64, n_kv_heads=8)
        out, kv = gqa(x)   # x: (B, T, D)
    """

    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        n_kv_heads: int   = 1,
        max_seq:    int   = 4096,
        dropout:    float = 0.0,
        use_rope:   bool  = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads  % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads     = n_heads
        self.n_kv_heads  = n_kv_heads
        self.n_groups    = n_heads // n_kv_heads
        self.d_head      = d_model // n_heads
        self.scale       = self.d_head ** -0.5
        self.use_rope    = use_rope

        self.q_proj = nn.Linear(d_model, n_heads    * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop   = nn.Dropout(dropout)

        if use_rope:
            from nanomind.longctx.rope import RotaryEmbedding
            self.rope = RotaryEmbedding(self.d_head, max_seq=max_seq)

    def forward(
        self,
        x:       torch.Tensor,
        past_kv: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        GQA forward pass.

        Args:
            x:       ``(B, T, D)`` input.
            past_kv: Optional ``(k_cache, v_cache)`` for generation.

        Returns:
            ``(output, (k, v))``
        """
        B, T, D = x.shape
        H, G    = self.n_heads, self.n_groups
        Hkv, Dh = self.n_kv_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)     # (B,H,T,Dh)
        K = self.k_proj(x).view(B, T, Hkv, Dh).transpose(1, 2)   # (B,Hkv,T,Dh)
        V = self.v_proj(x).view(B, T, Hkv, Dh).transpose(1, 2)

        if past_kv is not None:
            pk, pv = past_kv
            K = torch.cat([pk, K], dim=2)
            V = torch.cat([pv, V], dim=2)

        T_kv = K.shape[2]
        offset = T_kv - T

        if self.use_rope:
            Q, K = self.rope.apply(Q, K, seq_len=T, offset=offset)

        # Expand KV to match Q heads: repeat each KV head n_groups times
        K_exp = K.repeat_interleave(G, dim=1)    # (B, H, T_kv, Dh)
        V_exp = V.repeat_interleave(G, dim=1)

        scores  = (Q @ K_exp.transpose(-2, -1)) * self.scale   # (B,H,T,T_kv)
        # Causal mask
        mask    = torch.triu(torch.ones(T, T_kv, dtype=torch.bool), diagonal=1 + offset)
        scores  = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.drop(weights)
        out     = (weights @ V_exp).transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out), (K, V)

    @property
    def kv_cache_factor(self) -> float:
        """KV cache size relative to MHA (n_kv_heads / n_heads)."""
        return self.n_kv_heads / self.n_heads

    def to_dict(self) -> dict:
        return {
            "n_heads":     self.n_heads,
            "n_kv_heads":  self.n_kv_heads,
            "n_groups":    self.n_groups,
            "kv_reduction": f"{self.n_heads//self.n_kv_heads}x",
        }
''')
commit("feat: add GroupedQueryAttention (GQA/MQA) — n_kv_heads, RoPE integration, kv_cache_factor")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Chunked attention (FlashAttention-style IO)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/longctx/chunked.py", '''\
"""
nanomind/longctx/chunked.py — Chunked / Tiled Attention (FlashAttention-style).

## The Memory Problem with Standard Attention

Standard attention materialises the full N×N attention matrix:
  scores = QK^T / √d    ← (B, H, T, T) tensor — HUGE for large T!

For T=16384, H=32, B=1: 16384 × 16384 × 32 × 4 bytes = 32 GB just for scores!

## FlashAttention (Dao et al., 2022)

FlashAttention avoids materialising the full matrix using:
  1. Tiling: split Q, K, V into blocks that fit in SRAM
  2. Online softmax: compute numerically stable softmax incrementally
  3. IO-awareness: minimise HBM reads/writes

Memory: O(T) instead of O(T²)
Speed: 2-4× faster than standard PyTorch attention

This module implements the pure-Python version (FlashAttention algorithm)
without CUDA kernels. For production, use flash-attn library.

## Online Softmax (Milakov & Gimelshein, 2018)

The key mathematical identity:
  softmax([x1, x2]) = softmax([x1, x2]) regardless of split

Online update rule for accumulating attention outputs in chunks:
  m_new = max(m_old, max(block_scores))
  l_new = e^{m_old - m_new} × l_old + Σ e^{scores - m_new}
  O_new = (e^{m_old - m_new} × l_old × O_old + block_contribution) / l_new

References:
  Dao et al. (2022) FlashAttention: https://arxiv.org/abs/2205.14135
  Dao et al. (2023) FlashAttention-2: https://arxiv.org/abs/2307.08691
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkedAttention(nn.Module):
    """
    Chunked attention: compute attention in tiles to save memory.

    This is a pure-Python reference implementation of FlashAttention's
    tiling algorithm. Reduces peak memory from O(T²) to O(T × chunk).

    Args:
        d_model:    Model dimension.
        n_heads:    Number of attention heads.
        chunk_size: Tile/chunk size (smaller = less memory, more overhead).
        causal:     Apply causal masking.

    Example::

        ca  = ChunkedAttention(d_model=256, n_heads=8, chunk_size=64)
        out = ca(x)   # x: (B, T, D)  — O(T × chunk) peak memory
    """

    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        chunk_size: int  = 64,
        causal:     bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads    = n_heads
        self.d_head     = d_model // n_heads
        self.chunk_size = chunk_size
        self.causal     = causal
        self.scale      = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chunked attention forward.

        Args:
            x: ``(B, T, D)`` input.

        Returns:
            ``(B, T, D)`` output.
        """
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, Dh).permute(0, 2, 1, 3)  # (B,H,T,Dh)
        K = self.k_proj(x).view(B, T, H, Dh).permute(0, 2, 1, 3)
        V = self.v_proj(x).view(B, T, H, Dh).permute(0, 2, 1, 3)

        out = self._chunked_attention(Q, K, V)                      # (B,H,T,Dh)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.o_proj(out)

    def _chunked_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """
        Online softmax chunked attention.

        Processes Q in chunks, accumulating the output with online softmax.
        """
        B, H, T, Dh = Q.shape
        C           = self.chunk_size
        O           = torch.zeros_like(Q)           # output accumulator
        l           = torch.zeros(B, H, T, 1)       # log-sum-exp normaliser
        m           = torch.full((B, H, T, 1), float("-inf"))  # running max

        # Process K/V in chunks
        for kv_start in range(0, T, C):
            kv_end = min(kv_start + C, T)
            K_c    = K[:, :, kv_start:kv_end, :]    # (B,H,C,Dh)
            V_c    = V[:, :, kv_start:kv_end, :]

            # Scores for all Q against this K chunk
            scores = (Q @ K_c.transpose(-2, -1)) * self.scale  # (B,H,T,C)

            # Causal mask: Q position i cannot attend to K position j > i
            if self.causal:
                q_idx  = torch.arange(T).unsqueeze(-1)              # (T,1)
                kv_idx = torch.arange(kv_start, kv_end).unsqueeze(0)# (1,C)
                mask   = kv_idx > q_idx
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Online softmax update
            m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)
            exp_scores = torch.exp(scores - m_new)                  # (B,H,T,C)
            l_new = torch.exp(m - m_new) * l + exp_scores.sum(-1, keepdim=True)
            O     = (torch.exp(m - m_new) * l * O + exp_scores @ V_c) / (l_new + 1e-9)
            m     = m_new
            l     = l_new

        return O

    @property
    def peak_memory_ratio(self) -> float:
        """Peak memory relative to standard attention (chunk/T)."""
        return self.chunk_size  # lower is better
''')
commit("feat: add ChunkedAttention — tiled/chunked FlashAttention-style O(T×chunk) memory, online softmax")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — Long-context transformer
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/longctx/model.py", '''\
"""
nanomind/longctx/model.py — Long-Context Transformer LM.

Assembles a full language model using efficient attention mechanisms.
Configurable to use any combination of:
  - RoPE position encoding
  - ALiBi position bias
  - Grouped Query Attention (GQA/MQA)
  - Sliding Window Attention
  - Linear Attention
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.longctx.rope import RotaryEmbedding
from nanomind.longctx.alibi import ALiBi
from nanomind.longctx.gqa import GroupedQueryAttention
from nanomind.longctx.sliding_window import SlidingWindowAttention
from nanomind.longctx.linear_attn import LinearAttention


class LongContextConfig:
    """Configuration for a long-context transformer."""
    def __init__(
        self,
        vocab_size:   int   = 1000,
        d_model:      int   = 128,
        n_layers:     int   = 4,
        n_heads:      int   = 4,
        n_kv_heads:   int   = 4,
        max_seq:      int   = 512,
        window_size:  int   = 128,
        n_sinks:      int   = 4,
        attn_type:    str   = "gqa",    # gqa | sliding | linear
        pos_encoding: str   = "rope",   # rope | alibi | none
        dropout:      float = 0.0,
    ) -> None:
        self.vocab_size   = vocab_size
        self.d_model      = d_model
        self.n_layers     = n_layers
        self.n_heads      = n_heads
        self.n_kv_heads   = n_kv_heads
        self.max_seq      = max_seq
        self.window_size  = window_size
        self.n_sinks      = n_sinks
        self.attn_type    = attn_type
        self.pos_encoding = pos_encoding
        self.dropout      = dropout


class LongContextBlock(nn.Module):
    """Single transformer block with configurable attention."""

    def __init__(self, cfg: LongContextConfig) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.ln2  = nn.LayerNorm(cfg.d_model)

        # Attention
        if cfg.attn_type == "gqa":
            self.attn = GroupedQueryAttention(
                cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                max_seq=cfg.max_seq, use_rope=(cfg.pos_encoding == "rope")
            )
        elif cfg.attn_type == "sliding":
            self.attn = SlidingWindowAttention(
                cfg.d_model, cfg.n_heads, cfg.window_size, cfg.n_sinks
            )
        elif cfg.attn_type == "linear":
            self.attn = LinearAttention(cfg.d_model, cfg.n_heads)
        else:
            raise ValueError(f"Unknown attn_type: {cfg.attn_type!r}")

        self.attn_type = cfg.attn_type
        self.alibi = ALiBi(cfg.n_heads, cfg.max_seq) if cfg.pos_encoding == "alibi" else None

        # FFN
        d_ff = cfg.d_model * 4
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, d_ff), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        h  = self.ln1(x)
        if self.attn_type == "linear":
            h = self.attn(h)
            kv = None
        else:
            h, kv = self.attn(h)
        x = x + h
        # FFN
        x = x + self.ff(self.ln2(x))
        return x


class LongContextLM(nn.Module):
    """
    Long-Context Language Model.

    Supports GQA, sliding window, and linear attention variants.

    Args:
        cfg: :class:`LongContextConfig`.
    """

    def __init__(self, cfg: LongContextConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Position embeddings only for non-RoPE, non-ALiBi
        if cfg.pos_encoding == "none":
            self.pos_emb = nn.Embedding(cfg.max_seq, cfg.d_model)
        else:
            self.pos_emb = None
        self.blocks  = nn.ModuleList([LongContextBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets:   torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T    = input_ids.shape
        T       = min(T, self.cfg.max_seq)
        x       = self.tok_emb(input_ids[:, :T])
        if self.pos_emb is not None:
            x   = x + self.pos_emb(torch.arange(T))

        for block in self.blocks:
            x = block(x)

        x      = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets[:, :T].contiguous().view(-1),
            )
        return logits, loss

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def effective_context(self) -> int:
        """Effective context length."""
        if self.cfg.attn_type == "sliding":
            first_swa = next(
                b.attn for b in self.blocks
                if isinstance(b.attn, SlidingWindowAttention)
            )
            return first_swa.effective_context(self.cfg.n_layers)
        return self.cfg.max_seq
''')
commit("feat: add LongContextConfig, LongContextBlock, LongContextLM — GQA/sliding/linear attention LM")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — longctx __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/longctx/__init__.py", '''\
"""NanoMind Long-Context sub-package — Efficient Attention for long sequences.

Implements the full suite of long-context attention mechanisms:
  1. RotaryEmbedding    — RoPE with linear/NTK scaling, apply(q,k)
  2. ALiBi              — linear bias attention, no position embeddings
  3. SlidingWindowAttention — O(T×W) local+sink attention (Mistral-style)
  4. LinearAttention    — O(T) kernel attention (ELU feature map)
  5. RetNetDecay        — decayed linear attention (RetNet-style)
  6. GroupedQueryAttention — GQA/MQA with RoPE, kv_cache_factor
  7. ChunkedAttention   — FlashAttention-style tiled memory-efficient attention
  8. LongContextConfig  — LM configuration
  9. LongContextLM      — full LM with pluggable attention

Primary exports:
    - :class:`RotaryEmbedding`         — apply, extend, linear/NTK scaling
    - :class:`ALiBi`                   — bias, apply_to_scores, slopes
    - :class:`SlidingWindowAttention`  — window_mask, sinks, past_kv
    - :class:`LinearAttention`         — ELU/ReLU feature map, O(T)
    - :class:`RetNetDecay`             — per-head gamma decay
    - :class:`GroupedQueryAttention`   — GQA, MQA, RoPE, kv_cache_factor
    - :class:`ChunkedAttention`        — tiled attention, online softmax
    - :class:`LongContextConfig`       — vocab, attn_type, pos_encoding
    - :class:`LongContextLM`           — n_params, effective_context
"""

from nanomind.longctx.rope import RotaryEmbedding
from nanomind.longctx.alibi import ALiBi
from nanomind.longctx.sliding_window import SlidingWindowAttention
from nanomind.longctx.linear_attn import LinearAttention, RetNetDecay
from nanomind.longctx.gqa import GroupedQueryAttention
from nanomind.longctx.chunked import ChunkedAttention
from nanomind.longctx.model import LongContextConfig, LongContextLM

__all__ = [
    "RotaryEmbedding",
    "ALiBi",
    "SlidingWindowAttention",
    "LinearAttention", "RetNetDecay",
    "GroupedQueryAttention",
    "ChunkedAttention",
    "LongContextConfig", "LongContextLM",
]
''')
commit("refactor: export all long-context components from nanomind/longctx/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — example
# ══════════════════════════════════════════════════════════════════════════════
write("examples/longctx_demo.py", '''\
"""
examples/longctx_demo.py — NanoMind Long-Context & Efficient Attention demo.

Demonstrates:
  1. RoPE: rotary position embeddings, linear/NTK scaling
  2. ALiBi: linear attention bias, no positional encodings
  3. Sliding Window Attention: O(T×W) with attention sinks
  4. Linear Attention: O(T) kernel attention
  5. RetNetDecay: decayed linear attention
  6. Grouped Query Attention (GQA/MQA)
  7. Chunked Attention: FlashAttention-style memory efficiency
  8. LongContextLM: full LM with all variants

Usage:
    python examples/longctx_demo.py
"""
import math
import torch
from nanomind.longctx import (
    RotaryEmbedding, ALiBi, SlidingWindowAttention,
    LinearAttention, RetNetDecay, GroupedQueryAttention,
    ChunkedAttention, LongContextConfig, LongContextLM,
)

V = 64
print("=" * 60)
print("NanoMind Long-Context & Efficient Attention Demo")
print("=" * 60)

# ── RoPE ──────────────────────────────────────────────────────────────────────
print("\n── Rotary Position Embedding (RoPE) ──")
rope = RotaryEmbedding(dim=32, base=10000.0, max_seq=128)
q    = torch.randn(2, 4, 16, 32)   # (B, H, T, D_head)
k    = torch.randn(2, 4, 16, 32)
q_r, k_r = rope.apply(q, k, seq_len=16)
print(f"  Q_rotated shape: {tuple(q_r.shape)}")
# Verify: relative position property
print(f"  RoPE config: {rope.to_dict()}")
# NTK scaling for long contexts
rope_ntk = RotaryEmbedding(dim=32, max_seq=512, scale_factor=4.0, scaling_type="ntk")
q_r2, _ = rope_ntk.apply(q, k, seq_len=16)
print(f"  NTK-scaled output shape: {tuple(q_r2.shape)}")

# ── ALiBi ────────────────────────────────────────────────────────────────────
print("\n── ALiBi: Attention with Linear Biases ──")
alibi  = ALiBi(n_heads=4, max_seq=64)
bias   = alibi.bias(seq_len=16)
print(f"  Bias shape: {tuple(bias.shape)}")
print(f"  Slopes: {[round(s,4) for s in alibi.slopes.tolist()]}")
# Apply to dummy scores
scores = torch.randn(2, 4, 16, 16)
biased = alibi.apply_to_scores(scores, seq_len=16)
print(f"  Biased scores shape: {tuple(biased.shape)}")
print(f"  Causal mask applied: future positions = -inf? "
      f"{biased[0, 0, 0, 1].item() == float('-inf')}")

# ── Sliding Window ────────────────────────────────────────────────────────────
print("\n── Sliding Window Attention (Mistral-style) ──")
swa = SlidingWindowAttention(d_model=32, n_heads=4, window_size=8, n_sinks=2)
x   = torch.randn(2, 16, 32)
out, kv = swa(x)
print(f"  SWA output: {tuple(out.shape)}")
print(f"  Effective context (4 layers): {swa.effective_context(4)}")

# ── Linear Attention ──────────────────────────────────────────────────────────
print("\n── Linear Attention: O(T) complexity ──")
la  = LinearAttention(d_model=32, n_heads=4, feature="elu")
x   = torch.randn(2, 8, 32)
out = la(x)
print(f"  Linear attn output: {tuple(out.shape)}")
print(f"  Complexity: {la.complexity}")

# ── RetNetDecay ───────────────────────────────────────────────────────────────
print("\n── RetNet Decay Attention ──")
retnet = RetNetDecay(d_model=32, n_heads=4, gamma_min=0.9, gamma_max=0.999)
out    = retnet(x)
print(f"  RetNet output: {tuple(out.shape)}")
print(f"  Gamma range: [{retnet.gammas.min():.3f}, {retnet.gammas.max():.3f}]")

# ── GQA ───────────────────────────────────────────────────────────────────────
print("\n── Grouped Query Attention (GQA/MQA) ──")
# MHA: 4 Q heads, 4 KV heads
mha = GroupedQueryAttention(32, n_heads=4, n_kv_heads=4)
# GQA: 4 Q heads, 2 KV heads
gqa = GroupedQueryAttention(32, n_heads=4, n_kv_heads=2)
# MQA: 4 Q heads, 1 KV head
mqa = GroupedQueryAttention(32, n_heads=4, n_kv_heads=1)
x   = torch.randn(2, 8, 32)
with torch.no_grad():
    o_mha, _ = mha(x); o_gqa, _ = gqa(x); o_mqa, _ = mqa(x)
print(f"  MHA output: {tuple(o_mha.shape)}  KV factor: {mha.kv_cache_factor:.2f}")
print(f"  GQA output: {tuple(o_gqa.shape)}  KV factor: {gqa.kv_cache_factor:.2f}")
print(f"  MQA output: {tuple(o_mqa.shape)}  KV factor: {mqa.kv_cache_factor:.2f}")
print(f"  GQA info: {gqa.to_dict()}")

# ── Chunked Attention ──────────────────────────────────────────────────────────
print("\n── Chunked Attention (FlashAttention-style) ──")
ca  = ChunkedAttention(d_model=32, n_heads=4, chunk_size=4, causal=True)
x   = torch.randn(2, 16, 32)
out = ca(x)
print(f"  Chunked output: {tuple(out.shape)}")

# ── LongContextLM ─────────────────────────────────────────────────────────────
print("\n── LongContextLM ──")
for attn_type, pos_enc in [("gqa", "rope"), ("sliding", "alibi"), ("linear", "none")]:
    cfg = LongContextConfig(
        vocab_size=V, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
        max_seq=32, window_size=8, attn_type=attn_type, pos_encoding=pos_enc,
    )
    model = LongContextLM(cfg)
    ids   = torch.randint(0, V, (2, 12))
    with torch.no_grad():
        logits, _ = model(ids)
    eff_ctx = model.effective_context()
    print(f"  [{attn_type:8}+{pos_enc:5}] logits={tuple(logits.shape)} "
          f"params={model.n_params:,} eff_ctx={eff_ctx}")

print("\nLong-context demo complete!")
''')
commit("feat: add examples/longctx_demo.py — RoPE, ALiBi, SWA, linear attn, GQA, chunked, LongContextLM")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_longctx.py", '''\
"""tests/test_longctx.py — Tests for NanoMind long-context efficient attention."""
import math
import pytest
import torch
from nanomind.longctx import (
    RotaryEmbedding, ALiBi, SlidingWindowAttention,
    LinearAttention, RetNetDecay, GroupedQueryAttention,
    ChunkedAttention, LongContextConfig, LongContextLM,
)

D, V, H = 32, 16, 4


# ── RotaryEmbedding ───────────────────────────────────────────────────────────

class TestRoPE:
    def _rope(self, dim=D):
        return RotaryEmbedding(dim=dim, max_seq=64)

    def test_output_shape(self):
        rope = self._rope()
        q    = torch.randn(2, H, 8, D)
        k    = torch.randn(2, H, 8, D)
        q_r, k_r = rope.apply(q, k, seq_len=8)
        assert q_r.shape == q.shape
        assert k_r.shape == k.shape

    def test_different_positions_different_output(self):
        rope = self._rope()
        q = torch.randn(1, 1, 4, D)
        k = torch.randn(1, 1, 4, D)
        q1, _ = rope.apply(q, k, seq_len=4)
        q2, _ = rope.apply(q, k, seq_len=4, offset=4)
        assert not torch.allclose(q1, q2)

    def test_norm_preserved(self):
        rope = self._rope()
        q = torch.randn(2, H, 4, D)
        k = torch.randn(2, H, 4, D)
        q_r, _ = rope.apply(q, k, seq_len=4)
        # RoPE is a rotation — should preserve norm
        assert torch.allclose(q.norm(dim=-1), q_r.norm(dim=-1), atol=1e-5)

    def test_ntk_scaling(self):
        rope = RotaryEmbedding(D, scale_factor=4.0, scaling_type="ntk", max_seq=64)
        q = torch.randn(1, 1, 4, D)
        k = torch.randn(1, 1, 4, D)
        q_r, _ = rope.apply(q, k, seq_len=4)
        assert q_r.shape == q.shape

    def test_extend_cache(self):
        rope = RotaryEmbedding(D, max_seq=32)
        rope.extend(64)
        assert rope.max_seq == 64

    def test_to_dict_keys(self):
        d = self._rope().to_dict()
        for k in ("dim", "base", "max_seq"):
            assert k in d


# ── ALiBi ─────────────────────────────────────────────────────────────────────

class TestALiBi:
    def test_bias_shape(self):
        a    = ALiBi(n_heads=H, max_seq=32)
        bias = a.bias(seq_len=8)
        assert bias.shape == (H, 8, 8)

    def test_causal_masking(self):
        a    = ALiBi(n_heads=H)
        bias = a.bias(seq_len=8)
        # Upper triangle should be -inf
        for h in range(H):
            assert bias[h, 0, 1].item() == float("-inf")

    def test_diagonal_is_zero(self):
        a    = ALiBi(n_heads=H)
        bias = a.bias(seq_len=8)
        for h in range(H):
            assert bias[h, 5, 5].item() == 0.0

    def test_slopes_positive(self):
        a = ALiBi(n_heads=H)
        assert (a.slopes > 0).all()

    def test_apply_to_scores_shape(self):
        a      = ALiBi(n_heads=H)
        scores = torch.randn(2, H, 8, 8)
        out    = a.apply_to_scores(scores, seq_len=8)
        assert out.shape == scores.shape

    def test_n_slopes_equals_n_heads(self):
        for n in [4, 8, 12]:
            a = ALiBi(n_heads=n)
            assert len(a.slopes) == n


# ── SlidingWindowAttention ────────────────────────────────────────────────────

class TestSWA:
    def _swa(self, W=8, sinks=2):
        return SlidingWindowAttention(D, H, window_size=W, n_sinks=sinks)

    def test_output_shape(self):
        swa  = self._swa()
        x    = torch.randn(2, 16, D)
        out, kv = swa(x)
        assert out.shape == (2, 16, D)

    def test_kv_cache_shape(self):
        swa  = self._swa()
        x    = torch.randn(2, 8, D)
        _, (k, v) = swa(x)
        assert k.shape[2] == 8

    def test_gradient_flows(self):
        swa = self._swa()
        x   = torch.randn(2, 4, D, requires_grad=True)
        out, _ = swa(x)
        out.sum().backward()
        assert x.grad is not None

    def test_effective_context(self):
        swa = SlidingWindowAttention(D, H, window_size=64, n_sinks=4)
        assert swa.effective_context(8) == 8 * 64 + 4


# ── LinearAttention ───────────────────────────────────────────────────────────

class TestLinearAttention:
    def test_output_shape(self):
        la  = LinearAttention(D, H, feature="elu")
        x   = torch.randn(2, 6, D)
        assert la(x).shape == (2, 6, D)

    def test_relu_feature(self):
        la  = LinearAttention(D, H, feature="relu")
        x   = torch.randn(2, 4, D)
        assert la(x).shape == (2, 4, D)

    def test_gradient_flows(self):
        la = LinearAttention(D, H)
        x  = torch.randn(2, 4, D, requires_grad=True)
        la(x).sum().backward()
        assert x.grad is not None

    def test_complexity_string(self):
        la = LinearAttention(D, H)
        assert "linear" in la.complexity.lower() or "T" in la.complexity


# ── RetNetDecay ───────────────────────────────────────────────────────────────

class TestRetNetDecay:
    def test_output_shape(self):
        r = RetNetDecay(D, H)
        x = torch.randn(2, 6, D)
        assert r(x).shape == (2, 6, D)

    def test_gamma_range(self):
        r = RetNetDecay(D, H, gamma_min=0.8, gamma_max=0.99)
        assert r.gammas.min().item() >= 0.8
        assert r.gammas.max().item() <= 0.99


# ── GroupedQueryAttention ──────────────────────────────────────────────────────

class TestGQA:
    def test_mha_output_shape(self):
        gqa = GroupedQueryAttention(D, H, n_kv_heads=H, max_seq=32)
        x   = torch.randn(2, 8, D)
        out, kv = gqa(x)
        assert out.shape == (2, 8, D)

    def test_gqa_output_shape(self):
        gqa = GroupedQueryAttention(D, H, n_kv_heads=2, max_seq=32)
        x   = torch.randn(2, 8, D)
        out, _ = gqa(x)
        assert out.shape == (2, 8, D)

    def test_mqa_output_shape(self):
        gqa = GroupedQueryAttention(D, H, n_kv_heads=1, max_seq=32)
        x   = torch.randn(2, 8, D)
        out, _ = gqa(x)
        assert out.shape == (2, 8, D)

    def test_kv_cache_factor(self):
        gqa = GroupedQueryAttention(D, H, n_kv_heads=2)
        assert gqa.kv_cache_factor == 0.5

    def test_to_dict_keys(self):
        d = GroupedQueryAttention(D, H, n_kv_heads=2).to_dict()
        for k in ("n_heads", "n_kv_heads", "kv_reduction"):
            assert k in d

    def test_invalid_kv_heads_raises(self):
        with pytest.raises(AssertionError):
            GroupedQueryAttention(D, n_heads=4, n_kv_heads=3)


# ── ChunkedAttention ──────────────────────────────────────────────────────────

class TestChunkedAttention:
    def test_output_shape(self):
        ca  = ChunkedAttention(D, H, chunk_size=4)
        x   = torch.randn(2, 8, D)
        assert ca(x).shape == (2, 8, D)

    def test_causal_output_differs_from_noncausal(self):
        x   = torch.randn(2, 8, D)
        ca  = ChunkedAttention(D, H, chunk_size=4, causal=True)
        nc  = ChunkedAttention(D, H, chunk_size=4, causal=False)
        nc.load_state_dict(ca.state_dict())
        o1  = ca(x)
        o2  = nc(x)
        assert not torch.allclose(o1, o2, atol=1e-3)

    def test_gradient_flows(self):
        ca = ChunkedAttention(D, H, chunk_size=4)
        x  = torch.randn(2, 4, D, requires_grad=True)
        ca(x).sum().backward()
        assert x.grad is not None


# ── LongContextLM ─────────────────────────────────────────────────────────────

class TestLongContextLM:
    def _model(self, attn="gqa", pos="rope"):
        cfg = LongContextConfig(V, D, n_layers=1, n_heads=H, n_kv_heads=2,
                                 max_seq=16, window_size=8, attn_type=attn,
                                 pos_encoding=pos)
        return LongContextLM(cfg)

    def test_gqa_rope_logits(self):
        m   = self._model("gqa", "rope")
        ids = torch.randint(0, V, (2, 6))
        logits, _ = m(ids)
        assert logits.shape == (2, 6, V)

    def test_sliding_alibi_logits(self):
        m   = self._model("sliding", "alibi")
        ids = torch.randint(0, V, (2, 6))
        logits, _ = m(ids)
        assert logits.shape == (2, 6, V)

    def test_linear_none_logits(self):
        m   = self._model("linear", "none")
        ids = torch.randint(0, V, (2, 6))
        logits, _ = m(ids)
        assert logits.shape == (2, 6, V)

    def test_loss_scalar(self):
        m   = self._model()
        ids = torch.randint(0, V, (2, 6))
        _, loss = m(ids, ids)
        assert loss.shape == ()

    def test_n_params_positive(self):
        m = self._model()
        assert m.n_params > 0

    def test_gradient_flows(self):
        m    = self._model()
        ids  = torch.randint(0, V, (2, 4))
        _, l = m(ids, ids)
        l.backward()
        has_grad = any(p.grad is not None for p in m.parameters())
        assert has_grad
''')
commit("test: add full long-context test suite — RoPE, ALiBi, SWA, linear attn, GQA, chunked, LongContextLM")

# COMMITS 12-18
for title, body in [
    ("test: add RoPE linear scaling reduces high-freq test", '''
class TestRoPEScaling:
    def test_linear_scaling_changes_freqs(self):
        r1 = RotaryEmbedding(D, scale_factor=1.0, scaling_type="none", max_seq=32)
        r2 = RotaryEmbedding(D, scale_factor=4.0, scaling_type="linear", max_seq=32)
        q = torch.randn(1, 1, 4, D); k = torch.randn(1, 1, 4, D)
        q1, _ = r1.apply(q, k, seq_len=4)
        q2, _ = r2.apply(q, k, seq_len=4)
        assert not torch.allclose(q1, q2)
'''),
    ("test: add ALiBi caching returns same tensor test", '''
class TestALiBiCache:
    def test_bias_cached(self):
        a  = ALiBi(n_heads=H)
        b1 = a.bias(16)
        b2 = a.bias(16)
        assert b1 is b2   # should be same object from cache

    def test_different_lengths_different_bias(self):
        a  = ALiBi(n_heads=H)
        b1 = a.bias(8)
        b2 = a.bias(16)
        assert b1.shape != b2.shape
'''),
    ("test: add SWA past_kv generation test", '''
class TestSWAPastKV:
    def test_with_past_kv_generation(self):
        swa = SlidingWindowAttention(D, H, window_size=8, n_sinks=2)
        x1 = torch.randn(2, 4, D)
        out1, kv1 = swa(x1)
        x2 = torch.randn(2, 2, D)
        out2, kv2 = swa(x2, past_kv=kv1)
        assert out2.shape == (2, 2, D)
'''),
    ("test: add GQA past_kv accumulates test", '''
class TestGQAPastKV:
    def test_kv_accumulates(self):
        gqa = GroupedQueryAttention(D, H, n_kv_heads=2, max_seq=32)
        x   = torch.randn(2, 4, D)
        _, kv1 = gqa(x)
        k1, v1 = kv1
        assert k1.shape[2] == 4
        x2 = torch.randn(2, 2, D)
        _, kv2 = gqa(x2, past_kv=kv1)
        k2, _ = kv2
        assert k2.shape[2] == 6   # 4 + 2
'''),
    ("test: add ChunkedAttention chunk_size independence test", '''
class TestChunkSizeIndependence:
    def test_different_chunk_sizes_same_output(self):
        """Same weights, different chunk sizes should give same causal attn."""
        torch.manual_seed(0)
        ca4 = ChunkedAttention(D, H, chunk_size=4, causal=True)
        ca8 = ChunkedAttention(D, H, chunk_size=8, causal=True)
        ca8.load_state_dict(ca4.state_dict())
        x   = torch.randn(1, 8, D)
        with torch.no_grad():
            o4 = ca4(x)
            o8 = ca8(x)
        assert torch.allclose(o4, o8, atol=1e-4)
'''),
    ("test: add LinearAttention longer sequences test", '''
class TestLinearAttnLong:
    def test_long_sequence(self):
        la = LinearAttention(D, H)
        x  = torch.randn(1, 64, D)   # longer than training
        assert la(x).shape == (1, 64, D)
'''),
    ("test: add LongContextLM effective context sliding test", '''
class TestEffectiveContext:
    def test_sliding_window_effective_context(self):
        cfg = LongContextConfig(
            vocab_size=V, d_model=D, n_layers=4, n_heads=H, n_kv_heads=2,
            max_seq=32, window_size=16, attn_type="sliding", pos_encoding="none"
        )
        m   = LongContextLM(cfg)
        eff = m.effective_context()
        assert eff == 4 * 16 + 4   # n_layers × window + sinks

    def test_gqa_effective_context_is_max_seq(self):
        cfg = LongContextConfig(V, D, n_layers=2, n_heads=H, n_kv_heads=2,
                                 max_seq=32, attn_type="gqa", pos_encoding="rope")
        m   = LongContextLM(cfg)
        assert m.effective_context() == 32
'''),
]:
    src = read("tests/test_longctx.py")
    src += "\n" + body
    write("tests/test_longctx.py", src)
    commit(title)

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v4.5.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"4.4.0\"", "__version__ = \"4.5.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v4.5.0 — Long-Context & Efficient Attention release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `moe_v2`     | Advanced MoE++ — TopK/ExpertChoice/Hash routing, capacity, z-loss, SwiGLU |",
    "| `moe_v2`     | Advanced MoE++ — TopK/ExpertChoice/Hash routing, capacity, z-loss, SwiGLU |\n"
    "| `longctx`    | Long-Context — RoPE/ALiBi/GQA/SWA/LinearAttn/ChunkedAttn, RetNet, StreamingLLM |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [4.5.0] — 2024 — Long-Context & Efficient Attention\n\n### Added\n"
      "- `RotaryEmbedding` — RoPE with linear/NTK scaling, pre-computed cos/sin cache\n"
      "- `ALiBi` — per-head linear bias, causal slope matrix, cache\n"
      "- `SlidingWindowAttention` — O(T×W) local attention + attention sinks (Mistral/StreamingLLM)\n"
      "- `LinearAttention` — O(T) ELU/ReLU kernel attention, recurrent form\n"
      "- `RetNetDecay` — per-head gamma decay (RetNet-style)\n"
      "- `GroupedQueryAttention` — GQA/MQA with RoPE, kv_cache_factor, past_kv\n"
      "- `ChunkedAttention` — FlashAttention-style tiled memory-efficient attention\n"
      "- `LongContextConfig` — unified LM configuration\n"
      "- `LongContextLM` — full LM with pluggable attention, effective_context()\n"
      "- `examples/longctx_demo.py` — full long-context demo\n\n---\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v4.5.0, update README and CHANGELOG for Day 45 Long-Context Attention")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 45 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v4.5.0",
    "-m", "NanoMind v4.5.0 — Long-Context & Efficient Attention", check=False)
r = run("git", "push", "origin", "v4.5.0", check=False)
print("Tag v4.5.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 45 COMPLETE — v4.5.0 TAGGED! ===")
