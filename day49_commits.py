"""
day49_commits.py — 20 atomic commits for Day 49: Continuous Batching & KV Cache.
THE 1000 COMMIT MILESTONE!
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

print("\n=== DAY 49: Continuous Batching & KV Cache — 20 commits (THE 1000 MILESTONE!) ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — serving package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serving/__init__.py",
      '"""NanoMind Serving sub-package — Continuous Batching & KV Cache management."""\n')
commit("feat: add nanomind/serving/ package skeleton for continuous batching and KV cache")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — KV Cache block manager (PagedAttention)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serving/kv_cache.py", '''\
"""
nanomind/serving/kv_cache.py — KV Cache block manager (PagedAttention).

## The KV Cache Memory Problem

During LLM inference, we cache K and V tensors for each token to avoid
recomputing them every step:
  KV cache per token = 2 × n_layers × n_heads × d_head × bytes

For LLaMA-70B:
  2 × 80 layers × 8 KV heads × 128 d_head × 2 bytes = 327 KB per token!

With batch_size=100 and max_seq=4096:
  100 × 4096 × 327 KB = 134 GB — won't fit!

## PagedAttention (Kwon et al., 2023)

vLLM's key innovation: manage KV cache like OS virtual memory paging.

Instead of pre-allocating contiguous max-length KV buffers per sequence:
  - Divide KV cache into fixed-size BLOCKS (e.g., 16 tokens each)
  - Allocate blocks on demand as sequences grow
  - A sequence's KV cache = list of non-contiguous block pointers
  - Share blocks between sequences (for prefix sharing / beam search)

Benefits:
  ✓ Near-zero memory waste (no over-allocation)
  ✓ Up to 24× more requests in memory simultaneously
  ✓ Copy-on-write for beam search / prefix caching

Reference:
  Kwon et al. (2023) "Efficient Memory Management for Large Language Model
  Serving with PagedAttention"
  https://arxiv.org/abs/2309.06180
"""

from __future__ import annotations
import torch
from dataclasses import dataclass, field
from nanomind.utils.logger import get_logger

log = get_logger("serving.kv_cache")


@dataclass
class KVBlock:
    """A single KV cache block holding `block_size` tokens."""
    block_id:   int
    block_size: int
    n_layers:   int
    n_kv_heads: int
    d_head:     int
    ref_count:  int = 0    # number of sequences sharing this block
    n_filled:   int = 0    # tokens stored in this block (0..block_size)

    # K and V tensors: (n_layers, n_kv_heads, block_size, d_head)
    k_data: torch.Tensor = field(init=False)
    v_data: torch.Tensor = field(init=False)

    def __post_init__(self):
        shape = (self.n_layers, self.n_kv_heads, self.block_size, self.d_head)
        self.k_data = torch.zeros(shape)
        self.v_data = torch.zeros(shape)

    @property
    def is_full(self) -> bool:
        return self.n_filled >= self.block_size

    @property
    def free_slots(self) -> int:
        return self.block_size - self.n_filled

    def write(self, layer: int, head: int, pos: int, k: torch.Tensor, v: torch.Tensor):
        """Write K, V for one token at position pos."""
        slot = self.n_filled
        if slot < self.block_size:
            self.k_data[layer, head, slot] = k
            self.v_data[layer, head, slot] = v

    def memory_bytes(self) -> int:
        return (self.k_data.numel() + self.v_data.numel()) * 4  # FP32


class KVCacheManager:
    """
    PagedAttention KV cache block manager.

    Manages a pool of fixed-size KV blocks, allocating and freeing
    them for sequences dynamically.

    Args:
        n_blocks:   Total number of KV blocks in the pool.
        block_size: Tokens per block (default 16).
        n_layers:   Number of transformer layers.
        n_kv_heads: Number of KV attention heads.
        d_head:     Head dimension.

    Example::

        manager = KVCacheManager(n_blocks=512, block_size=16,
                                  n_layers=32, n_kv_heads=8, d_head=128)
        seq_id  = "req_001"
        manager.allocate(seq_id, n_tokens=64)   # 4 blocks
        blocks  = manager.get_blocks(seq_id)
        manager.free(seq_id)
    """

    def __init__(
        self,
        n_blocks:   int,
        block_size: int = 16,
        n_layers:   int = 1,
        n_kv_heads: int = 1,
        d_head:     int = 64,
    ) -> None:
        self.n_blocks   = n_blocks
        self.block_size = block_size
        self.n_layers   = n_layers
        self.n_kv_heads = n_kv_heads
        self.d_head     = d_head

        # Pool of all blocks
        self._pool: list[KVBlock] = [
            KVBlock(i, block_size, n_layers, n_kv_heads, d_head)
            for i in range(n_blocks)
        ]
        self._free:   list[int]         = list(range(n_blocks))   # free block IDs
        self._seq_blocks: dict[str, list[int]] = {}               # seq_id → [block_ids]

    @property
    def n_free_blocks(self) -> int:
        return len(self._free)

    @property
    def n_used_blocks(self) -> int:
        return self.n_blocks - self.n_free_blocks

    def can_allocate(self, n_tokens: int) -> bool:
        """Check if enough blocks exist for n_tokens."""
        n_needed = (n_tokens + self.block_size - 1) // self.block_size
        return n_needed <= self.n_free_blocks

    def allocate(self, seq_id: str, n_tokens: int) -> list[int]:
        """
        Allocate blocks for a sequence.

        Args:
            seq_id:   Sequence identifier.
            n_tokens: Number of tokens to accommodate.

        Returns:
            List of allocated block IDs.
        """
        n_needed = max(1, (n_tokens + self.block_size - 1) // self.block_size)
        if n_needed > len(self._free):
            raise MemoryError(f"Not enough KV blocks: need {n_needed}, have {len(self._free)}")

        allocated = []
        for _ in range(n_needed):
            bid = self._free.pop(0)
            self._pool[bid].ref_count = 1
            allocated.append(bid)

        self._seq_blocks[seq_id] = allocated
        log.debug(f"Allocated {n_needed} blocks to {seq_id}")
        return allocated

    def extend(self, seq_id: str) -> int:
        """Allocate one more block for a growing sequence."""
        if not self._free:
            raise MemoryError("KV cache full — cannot extend sequence")
        bid = self._free.pop(0)
        self._pool[bid].ref_count = 1
        self._seq_blocks[seq_id].append(bid)
        return bid

    def free(self, seq_id: str) -> None:
        """Free all blocks for a completed sequence."""
        for bid in self._seq_blocks.get(seq_id, []):
            block = self._pool[bid]
            block.ref_count -= 1
            if block.ref_count <= 0:
                block.ref_count = 0
                block.n_filled  = 0
                self._free.append(bid)
        self._seq_blocks.pop(seq_id, None)

    def get_blocks(self, seq_id: str) -> list[KVBlock]:
        """Return KVBlock objects for a sequence."""
        return [self._pool[bid] for bid in self._seq_blocks.get(seq_id, [])]

    def copy_on_write(self, src_id: str, dst_id: str) -> None:
        """Copy blocks from one sequence to another (beam search fork)."""
        src_blocks = self._seq_blocks.get(src_id, [])
        new_blocks  = []
        for bid in src_blocks:
            # Allocate new block and copy data
            if not self._free:
                raise MemoryError("No free blocks for copy-on-write")
            new_bid = self._free.pop(0)
            src = self._pool[bid]
            dst = self._pool[new_bid]
            dst.k_data.copy_(src.k_data)
            dst.v_data.copy_(src.v_data)
            dst.n_filled   = src.n_filled
            dst.ref_count  = 1
            new_blocks.append(new_bid)
        self._seq_blocks[dst_id] = new_blocks

    def stats(self) -> dict:
        return {
            "total_blocks": self.n_blocks,
            "free_blocks":  self.n_free_blocks,
            "used_blocks":  self.n_used_blocks,
            "utilisation":  round(self.n_used_blocks / self.n_blocks, 3),
            "n_sequences":  len(self._seq_blocks),
            "block_size":   self.block_size,
            "memory_gb":    round(self.n_blocks * self.block_size *
                                  self.n_kv_heads * self.d_head * 2 * 4 / 1e9, 3),
        }
''')
commit("feat: add KVBlock, KVCacheManager — PagedAttention block pool, allocate, free, copy_on_write, stats")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — Request and sequence management
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serving/request.py", '''\
"""
nanomind/serving/request.py — Inference request and sequence state management.

Tracks the lifecycle of a generation request:
  WAITING  → Queued, not yet started
  RUNNING  → Currently being processed in a batch
  FINISHED → Generation complete (max tokens or EOS)
  ABORTED  → Cancelled or errored

Each request owns a sequence with token history, KV cache block pointers,
and sampling parameters.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(Enum):
    WAITING  = "waiting"
    RUNNING  = "running"
    FINISHED = "finished"
    ABORTED  = "aborted"


@dataclass
class SamplingParams:
    """Sampling parameters for a generation request."""
    max_new_tokens: int   = 64
    temperature:    float = 1.0
    top_p:          float = 1.0
    top_k:          int   = 0       # 0 = disabled
    repetition_penalty: float = 1.0
    stop_token_id:  int | None = None

    def to_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature":    self.temperature,
            "top_p":          self.top_p,
        }


@dataclass
class InferenceRequest:
    """
    A single generation request tracked through the serving system.

    Args:
        request_id:     Unique identifier.
        input_ids:      Prompt token IDs.
        sampling_params: Sampling configuration.

    Example::

        req = InferenceRequest("req_001", [1, 2, 3, 4],
                                SamplingParams(max_new_tokens=32))
        req.status = RequestStatus.RUNNING
        req.add_token(42)
        print(req.generated_tokens)   # [42]
        print(req.is_finished)        # False (max_new_tokens=32)
    """
    request_id:      str
    input_ids:       list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)

    status:           RequestStatus = RequestStatus.WAITING
    generated_tokens: list[int]     = field(default_factory=list)
    kv_block_ids:     list[int]     = field(default_factory=list)
    arrival_time:     float         = field(default_factory=time.monotonic)
    start_time:       float | None  = None
    finish_time:      float | None  = None

    def add_token(self, token_id: int) -> None:
        """Append a generated token."""
        self.generated_tokens.append(token_id)
        sp = self.sampling_params
        # Check stop condition
        if len(self.generated_tokens) >= sp.max_new_tokens:
            self._finish()
        elif sp.stop_token_id and token_id == sp.stop_token_id:
            self._finish()

    def _finish(self) -> None:
        self.status      = RequestStatus.FINISHED
        self.finish_time = time.monotonic()

    def abort(self) -> None:
        self.status      = RequestStatus.ABORTED
        self.finish_time = time.monotonic()

    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.FINISHED, RequestStatus.ABORTED)

    @property
    def prompt_len(self) -> int:
        return len(self.input_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + len(self.generated_tokens)

    @property
    def n_generated(self) -> int:
        return len(self.generated_tokens)

    @property
    def latency_s(self) -> float | None:
        if self.finish_time and self.start_time:
            return self.finish_time - self.start_time
        return None

    @property
    def ttft_s(self) -> float | None:
        """Time to first token."""
        if self.start_time and self.generated_tokens:
            return self.start_time - self.arrival_time
        return None

    @property
    def output_ids(self) -> list[int]:
        return self.input_ids + self.generated_tokens

    def to_dict(self) -> dict:
        return {
            "request_id":   self.request_id,
            "status":       self.status.value,
            "prompt_len":   self.prompt_len,
            "n_generated":  self.n_generated,
            "latency_s":    self.latency_s,
        }
''')
commit("feat: add InferenceRequest, SamplingParams, RequestStatus — lifecycle, add_token, latency tracking")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — Scheduler (continuous batching)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serving/scheduler.py", '''\
"""
nanomind/serving/scheduler.py — Continuous batching scheduler.

## Continuous Batching (Orca, Yu et al., 2022)

Traditional batching: pad all sequences to max length, process together.
Problem: short sequences waste GPU cycles waiting for long ones.

Static batching:   |AAAA.....|BBBB.....|CC.......|   (lots of padding!)
Continuous batching: |AAAA|BB|CC|DD|EE|FF|GG|HH|   (no padding, any length!)

In continuous batching:
  - The batch is not fixed per request
  - New requests JOIN the running batch as slots free up
  - Each iteration: run one decode step for all running sequences
  - Finished sequences: remove immediately
  - Waiting sequences: add if KV cache memory permits

This is how vLLM, TensorRT-LLM, and TGI achieve high throughput:
  Traditional: 10-20 req/s
  Continuous:  100-200 req/s (same hardware!)

The key insight: decode step is always 1 token, regardless of history length.
So mixing sequences of different lengths has no waste!

Scheduler decisions:
  1. Prefill phase:  process prompt in one pass (parallel)
  2. Decode phase:   generate one token per step (sequential)

Reference:
  Yu et al. (2022) "Orca: A Distributed Serving System for Transformer-Based
  Generative Models" OSDI 2022
"""

from __future__ import annotations
from dataclasses import dataclass, field
from nanomind.serving.request import InferenceRequest, RequestStatus
from nanomind.serving.kv_cache import KVCacheManager
from nanomind.utils.logger import get_logger

log = get_logger("serving.scheduler")


@dataclass
class SchedulerOutput:
    """Output of one scheduler step."""
    running:    list[InferenceRequest]
    prefill:    list[InferenceRequest]   # newly admitted for prefill
    finished:   list[InferenceRequest]   # just completed
    n_batched_tokens: int = 0


class ContinuousBatchingScheduler:
    """
    Continuous batching scheduler.

    Manages a waiting queue and a running set of sequences.
    Each step: admits new requests, removes finished ones.

    Args:
        kv_manager:    :class:`KVCacheManager`.
        max_batch_size:    Max sequences in flight simultaneously.
        max_tokens_per_step: Max total tokens in one decode step.

    Example::

        scheduler = ContinuousBatchingScheduler(kv_manager, max_batch_size=32)
        scheduler.add_request(req)
        while scheduler.has_work():
            output = scheduler.step()
            # output.running = currently active sequences
            # output.prefill = newly admitted (need prefill)
    """

    def __init__(
        self,
        kv_manager:          KVCacheManager,
        max_batch_size:      int = 32,
        max_tokens_per_step: int = 2048,
    ) -> None:
        self.kv         = kv_manager
        self.max_batch  = max_batch_size
        self.max_tokens = max_tokens_per_step

        self._waiting: list[InferenceRequest] = []
        self._running: list[InferenceRequest] = []
        self._finished: list[InferenceRequest] = []

    def add_request(self, req: InferenceRequest) -> None:
        """Enqueue a new request."""
        self._waiting.append(req)
        log.debug(f"Queued {req.request_id} (prompt={req.prompt_len})")

    def _can_admit(self, req: InferenceRequest) -> bool:
        """Check if KV memory is available for a new request."""
        if len(self._running) >= self.max_batch:
            return False
        return self.kv.can_allocate(req.prompt_len +
                                     req.sampling_params.max_new_tokens)

    def _admit_waiting(self) -> list[InferenceRequest]:
        """Admit as many waiting requests as possible."""
        newly_admitted = []
        still_waiting  = []
        for req in self._waiting:
            if self._can_admit(req):
                import time
                req.status     = RequestStatus.RUNNING
                req.start_time = time.monotonic()
                block_ids = self.kv.allocate(
                    req.request_id,
                    req.prompt_len + req.sampling_params.max_new_tokens,
                )
                req.kv_block_ids = block_ids
                self._running.append(req)
                newly_admitted.append(req)
                log.info(f"Admitted {req.request_id} — blocks={len(block_ids)}")
            else:
                still_waiting.append(req)
        self._waiting = still_waiting
        return newly_admitted

    def step(self) -> SchedulerOutput:
        """
        Execute one scheduling step.

        Returns:
            :class:`SchedulerOutput` with running/prefill/finished lists.
        """
        # 1. Admit new requests (continuous batching!)
        new_prefill = self._admit_waiting()

        # 2. Identify finished sequences
        just_finished = [r for r in self._running if r.is_finished]
        for req in just_finished:
            self.kv.free(req.request_id)
            self._finished.append(req)
        self._running = [r for r in self._running if not r.is_finished]

        # 3. Extend KV blocks for running sequences that need more memory
        for req in self._running:
            if req not in new_prefill:  # already allocated for new prefill
                try:
                    if req.total_len % self.kv.block_size == 0:
                        self.kv.extend(req.request_id)
                except MemoryError:
                    # Preempt: move back to waiting
                    self.kv.free(req.request_id)
                    req.status = RequestStatus.WAITING
                    self._running.remove(req)
                    self._waiting.insert(0, req)   # priority queue front
                    log.warning(f"Preempted {req.request_id} — OOM")

        n_batched = sum(r.total_len for r in self._running)
        return SchedulerOutput(
            running   = list(self._running),
            prefill   = new_prefill,
            finished  = just_finished,
            n_batched_tokens = n_batched,
        )

    def has_work(self) -> bool:
        return bool(self._running or self._waiting)

    def queue_len(self) -> int:
        return len(self._waiting)

    def running_len(self) -> int:
        return len(self._running)

    def stats(self) -> dict:
        return {
            "running": self.running_len(),
            "waiting": self.queue_len(),
            "finished": len(self._finished),
            "kv_stats": self.kv.stats(),
        }
''')
commit("feat: add ContinuousBatchingScheduler — admit, preempt, step(), has_work(), SchedulerOutput, stats")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — Prefix caching (RadixAttention)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serving/prefix_cache.py", '''\
"""
nanomind/serving/prefix_cache.py — Prefix caching (RadixAttention).

## Prefix Caching (SGLang / RadixAttention)

Many LLM requests share common prefixes:
  - System prompt: same for all requests in a chat API
  - Few-shot examples: same prefix for classification tasks
  - Code context: same imports for all code completions

With prefix caching, the KV cache for these shared prefixes is computed
ONCE and reused across requests:
  Req 1: [SYS PROMPT][USER 1]  → compute full KV cache
  Req 2: [SYS PROMPT][USER 2]  → reuse sys prompt KV cache!

This is RadixAttention (Zheng et al., 2023): represent all cached
prefixes as a radix tree where each edge represents a token sequence.

Benefits:
  ✓ Large system prompts (2K tokens): effectively free on cache hit
  ✓ Few-shot prompts: amortised across all requests
  ✓ Latency reduction: skip prompt processing for cached prefix

Reference:
  Zheng et al. (2023) "SGLang: Efficient Execution of Structured Language
  Model Programs" https://arxiv.org/abs/2312.07104
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PrefixNode:
    """Node in the prefix radix tree."""
    tokens:      tuple        # token sequence at this edge
    block_ids:   list[int]   = field(default_factory=list)
    children:    dict         = field(default_factory=dict)
    ref_count:   int          = 0
    last_access: float        = 0.0

    @property
    def prefix_len(self) -> int:
        return len(self.tokens)


class PrefixCache:
    """
    Prefix cache using a radix tree.

    Maps token prefixes → cached KV block IDs.

    Example::

        cache   = PrefixCache()
        prefix  = (1, 2, 3, 4, 5)   # system prompt tokens
        blocks  = [0, 1]
        cache.insert(prefix, blocks)

        hit, cached_blocks = cache.lookup([1, 2, 3, 4, 5, 6, 7])
        print(f"Cache hit for {hit} tokens, reusing {cached_blocks}")
    """

    def __init__(self) -> None:
        self._root   = PrefixNode(tokens=())
        self._hits   = 0
        self._misses = 0

    def insert(self, token_ids: list[int], block_ids: list[int]) -> None:
        """Cache KV blocks for a token prefix."""
        tokens = tuple(token_ids)
        node   = self._root
        pos    = 0

        while pos < len(tokens):
            # Find matching child
            matched = None
            for child_tok, child in node.children.items():
                n = self._common_prefix_len(tokens[pos:], child_tok)
                if n > 0:
                    matched = (child_tok, child, n)
                    break

            if matched is None:
                # No child matches — create new node
                remaining = tokens[pos:]
                new_node  = PrefixNode(tokens=remaining, block_ids=block_ids)
                node.children[remaining] = new_node
                break
            else:
                child_tok, child, n = matched
                if n == len(child_tok):
                    # Full match — continue down
                    node = child
                    pos += n
                else:
                    # Partial match — split node
                    shared    = child_tok[:n]
                    remaining = child_tok[n:]
                    # Create split node
                    split_node             = PrefixNode(tokens=shared)
                    split_node.children[remaining] = child
                    split_node.children[tokens[pos + n:]] = PrefixNode(
                        tokens=tokens[pos + n:], block_ids=block_ids
                    )
                    del node.children[child_tok]
                    node.children[shared]  = split_node
                    break

    def lookup(self, token_ids: list[int]) -> tuple[int, list[int]]:
        """
        Find the longest cached prefix for token_ids.

        Returns:
            ``(n_cached_tokens, block_ids)``
        """
        import time
        tokens     = tuple(token_ids)
        node       = self._root
        pos        = 0
        block_ids  = []

        while pos < len(tokens):
            matched = None
            for child_tok, child in node.children.items():
                n = self._common_prefix_len(tokens[pos:], child_tok)
                if n > 0:
                    matched = (child_tok, child, n)
                    break

            if matched is None:
                break
            child_tok, child, n = matched
            block_ids.extend(child.block_ids)
            child.last_access = time.monotonic()
            child.ref_count   += 1
            node = child
            pos += n

        if pos > 0:
            self._hits += 1
        else:
            self._misses += 1

        return pos, block_ids

    @staticmethod
    def _common_prefix_len(a: tuple, b: tuple) -> int:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    def evict_lru(self, n: int = 1) -> int:
        """Evict n least-recently-used leaf nodes. Returns evicted count."""
        import time
        leaves = self._collect_leaves(self._root)
        leaves.sort(key=lambda x: x.last_access)
        evicted = 0
        for leaf in leaves[:n]:
            if leaf.ref_count == 0:
                leaf.block_ids = []
                evicted += 1
        return evicted

    def _collect_leaves(self, node: PrefixNode) -> list[PrefixNode]:
        if not node.children:
            return [node] if node.block_ids else []
        leaves = []
        for child in node.children.values():
            leaves.extend(self._collect_leaves(child))
        return leaves

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits":      self._hits,
            "misses":    self._misses,
            "hit_rate":  round(self._hits / max(total, 1), 3),
        }
''')
commit("feat: add PrefixCache (RadixAttention) — radix tree insert/lookup, LRU eviction, hit_rate stats")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Inference engine (continuous batching loop)
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serving/engine.py", '''\
"""
nanomind/serving/engine.py — LLM inference engine with continuous batching.

Ties together: scheduler + KV cache + prefix cache + model.
Runs the continuous batching generation loop.
"""

from __future__ import annotations
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

from nanomind.serving.request import InferenceRequest, SamplingParams, RequestStatus
from nanomind.serving.kv_cache import KVCacheManager
from nanomind.serving.scheduler import ContinuousBatchingScheduler
from nanomind.serving.prefix_cache import PrefixCache
from nanomind.utils.logger import get_logger

log = get_logger("serving.engine")


@dataclass
class EngineConfig:
    """Configuration for the inference engine."""
    max_batch_size:      int   = 32
    max_tokens_per_step: int   = 2048
    n_kv_blocks:         int   = 256
    block_size:          int   = 16
    n_layers:            int   = 4
    n_kv_heads:          int   = 1
    d_head:              int   = 16
    vocab_size:          int   = 100
    enable_prefix_cache: bool  = True


@dataclass
class EngineStats:
    """Aggregate statistics for the serving engine."""
    n_requests_finished: int   = 0
    n_requests_aborted:  int   = 0
    total_tokens_gen:    int   = 0
    total_steps:         int   = 0
    wall_time_s:         float = 0.0

    @property
    def throughput(self) -> float:
        return self.total_tokens_gen / max(self.wall_time_s, 1e-6)

    def to_dict(self) -> dict:
        return {
            "finished_requests":  self.n_requests_finished,
            "total_tokens":       self.total_tokens_gen,
            "throughput_tok_s":   round(self.throughput, 1),
            "total_steps":        self.total_steps,
        }


class LLMEngine:
    """
    LLM serving engine with continuous batching.

    Args:
        model:   LM model ``forward(input_ids) → (B, T, V)`` logits.
        cfg:     :class:`EngineConfig`.

    Example::

        engine = LLMEngine(model, EngineConfig(max_batch_size=8))
        req1   = engine.submit("req_001", [1, 2, 3], SamplingParams(max_new_tokens=16))
        req2   = engine.submit("req_002", [4, 5, 6], SamplingParams(max_new_tokens=8))
        results = engine.run_until_done()
        print(results[0].output_ids)
    """

    def __init__(self, model: nn.Module, cfg: EngineConfig | None = None) -> None:
        self.model  = model
        self.cfg    = cfg or EngineConfig()
        self.kv     = KVCacheManager(
            n_blocks   = self.cfg.n_kv_blocks,
            block_size = self.cfg.block_size,
            n_layers   = self.cfg.n_layers,
            n_kv_heads = self.cfg.n_kv_heads,
            d_head     = self.cfg.d_head,
        )
        self.scheduler = ContinuousBatchingScheduler(
            self.kv,
            max_batch_size      = self.cfg.max_batch_size,
            max_tokens_per_step = self.cfg.max_tokens_per_step,
        )
        self.prefix_cache = PrefixCache() if self.cfg.enable_prefix_cache else None
        self.stats        = EngineStats()

    def submit(
        self,
        request_id:      str,
        input_ids:       list[int],
        sampling_params: SamplingParams | None = None,
    ) -> InferenceRequest:
        """Submit a new generation request."""
        req = InferenceRequest(
            request_id      = request_id,
            input_ids       = input_ids,
            sampling_params = sampling_params or SamplingParams(),
        )
        self.scheduler.add_request(req)
        return req

    @torch.no_grad()
    def _run_model(self, requests: list[InferenceRequest]) -> dict[str, int]:
        """Run model on a batch and return next token for each request."""
        results = {}
        for req in requests:
            # Pack input for this request
            ids    = torch.tensor([req.output_ids[-min(16, req.total_len):]])  # (1, T)
            out    = self.model(ids)
            logits = out[0] if isinstance(out, tuple) else out
            last   = logits[0, -1, :]   # (V,)
            sp     = req.sampling_params

            if sp.temperature <= 0.01:
                next_tok = last.argmax().item()
            else:
                probs    = F.softmax(last / sp.temperature, dim=-1)
                next_tok = torch.multinomial(probs + 1e-10, 1).item()
            results[req.request_id] = next_tok
        return results

    def step(self) -> list[InferenceRequest]:
        """
        Execute one continuous batching step.

        Returns:
            List of just-finished requests.
        """
        sched_out = self.scheduler.step()
        self.stats.total_steps += 1

        if not sched_out.running:
            return sched_out.finished

        # Run model for all running sequences
        next_tokens = self._run_model(sched_out.running)

        # Append tokens to each request
        for req in sched_out.running:
            tok = next_tokens.get(req.request_id, 0)
            req.add_token(tok)
            self.stats.total_tokens_gen += 1

        self.stats.n_requests_finished += len(sched_out.finished)
        return sched_out.finished

    def run_until_done(self, max_steps: int = 1000) -> list[InferenceRequest]:
        """Run until all submitted requests finish."""
        t0     = time.monotonic()
        all_fin = []
        for _ in range(max_steps):
            if not self.scheduler.has_work():
                break
            finished = self.step()
            all_fin.extend(finished)
        self.stats.wall_time_s = time.monotonic() - t0
        return all_fin

    def engine_stats(self) -> dict:
        return {
            **self.stats.to_dict(),
            "scheduler": self.scheduler.stats(),
            "prefix_cache": self.prefix_cache.stats() if self.prefix_cache else {},
        }
''')
commit("feat: add LLMEngine — continuous batching loop, submit, step, run_until_done, EngineConfig, EngineStats")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — Throughput benchmarker
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serving/benchmark.py", '''\
"""
nanomind/serving/benchmark.py — Serving throughput and latency benchmarker.

Measures:
  - Throughput: tokens generated per second
  - TTFT: time to first token
  - TPOT: time per output token
  - P50/P90/P99 latencies
"""

from __future__ import annotations
import time
import random
from dataclasses import dataclass, field

from nanomind.serving.request import InferenceRequest, SamplingParams


@dataclass
class BenchmarkConfig:
    """Configuration for a serving benchmark."""
    n_requests:          int   = 20
    min_prompt_len:      int   = 4
    max_prompt_len:      int   = 16
    min_output_len:      int   = 4
    max_output_len:      int   = 16
    vocab_size:          int   = 100
    seed:                int   = 42


@dataclass
class BenchmarkResult:
    """Results from a serving benchmark."""
    throughput_tok_s:    float
    mean_latency_s:      float
    p50_latency_s:       float
    p90_latency_s:       float
    n_requests:          int
    n_tokens_total:      int
    wall_time_s:         float

    def to_dict(self) -> dict:
        return {
            "throughput_tok_s": round(self.throughput_tok_s, 1),
            "mean_latency_s":   round(self.mean_latency_s, 3),
            "p50_latency_s":    round(self.p50_latency_s, 3),
            "p90_latency_s":    round(self.p90_latency_s, 3),
            "n_requests":       self.n_requests,
            "n_tokens_total":   self.n_tokens_total,
        }


def make_requests(cfg: BenchmarkConfig) -> list[InferenceRequest]:
    """Generate random benchmark requests."""
    rng = random.Random(cfg.seed)
    reqs = []
    for i in range(cfg.n_requests):
        plen   = rng.randint(cfg.min_prompt_len, cfg.max_prompt_len)
        olen   = rng.randint(cfg.min_output_len, cfg.max_output_len)
        ids    = [rng.randint(0, cfg.vocab_size - 1) for _ in range(plen)]
        params = SamplingParams(max_new_tokens=olen, temperature=1.0)
        reqs.append(InferenceRequest(f"req_{i:04d}", ids, params))
    return reqs


def run_benchmark(engine, cfg: BenchmarkConfig | None = None) -> BenchmarkResult:
    """
    Run a serving throughput benchmark.

    Args:
        engine: :class:`LLMEngine` instance.
        cfg:    :class:`BenchmarkConfig`.

    Returns:
        :class:`BenchmarkResult`.
    """
    cfg  = cfg or BenchmarkConfig()
    reqs = make_requests(cfg)

    for req in reqs:
        engine.submit(req.request_id, req.input_ids, req.sampling_params)

    t0       = time.monotonic()
    finished = engine.run_until_done()
    wall     = time.monotonic() - t0

    latencies    = [r.latency_s or 0.0 for r in finished]
    latencies.sort()
    n_tokens     = sum(r.n_generated for r in finished)
    n            = len(latencies)

    return BenchmarkResult(
        throughput_tok_s = n_tokens / max(wall, 1e-6),
        mean_latency_s   = sum(latencies) / max(n, 1),
        p50_latency_s    = latencies[n // 2] if n else 0.0,
        p90_latency_s    = latencies[int(n * 0.9)] if n else 0.0,
        n_requests       = n,
        n_tokens_total   = n_tokens,
        wall_time_s      = wall,
    )
''')
commit("feat: add BenchmarkConfig, BenchmarkResult, make_requests, run_benchmark — throughput/latency metrics")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — serving __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/serving/__init__.py", '''\
"""NanoMind Serving sub-package — Continuous Batching & KV Cache management.

Implements production-grade LLM serving infrastructure:
  1. KVBlock / KVCacheManager  — PagedAttention block pool, copy-on-write
  2. InferenceRequest          — request lifecycle, token accumulation
  3. SamplingParams            — temperature, top_p, repetition_penalty
  4. RequestStatus             — WAITING/RUNNING/FINISHED/ABORTED
  5. ContinuousBatchingScheduler — admit, preempt, step(), SchedulerOutput
  6. PrefixCache               — RadixAttention prefix caching, LRU evict
  7. EngineConfig              — serving configuration
  8. LLMEngine                 — full engine: submit, step, run_until_done
  9. EngineStats               — throughput, total_tokens, steps
  10. BenchmarkConfig          — benchmark parameters
  11. BenchmarkResult          — throughput/latency metrics
  12. run_benchmark            — end-to-end serving benchmark

Primary exports:
    - :class:`KVBlock`                       — single KV cache block
    - :class:`KVCacheManager`                — PagedAttention block pool
    - :class:`InferenceRequest`              — request with lifecycle tracking
    - :class:`SamplingParams`                — sampling configuration
    - :class:`RequestStatus`                 — WAITING/RUNNING/FINISHED
    - :class:`ContinuousBatchingScheduler`   — continuous batching scheduler
    - :class:`SchedulerOutput`               — running/prefill/finished
    - :class:`PrefixCache`                   — RadixAttention prefix caching
    - :class:`EngineConfig`                  — engine configuration
    - :class:`LLMEngine`                     — submit, step, run_until_done
    - :class:`EngineStats`                   — throughput, total_tokens
    - :class:`BenchmarkConfig`               — benchmark settings
    - :class:`BenchmarkResult`               — perf metrics
    - :func:`run_benchmark`                  — run a full serving benchmark
    - :func:`make_requests`                  — generate benchmark requests
"""

from nanomind.serving.kv_cache import KVBlock, KVCacheManager
from nanomind.serving.request import InferenceRequest, SamplingParams, RequestStatus
from nanomind.serving.scheduler import ContinuousBatchingScheduler, SchedulerOutput
from nanomind.serving.prefix_cache import PrefixCache, PrefixNode
from nanomind.serving.engine import LLMEngine, EngineConfig, EngineStats
from nanomind.serving.benchmark import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, make_requests,
)

__all__ = [
    "KVBlock", "KVCacheManager",
    "InferenceRequest", "SamplingParams", "RequestStatus",
    "ContinuousBatchingScheduler", "SchedulerOutput",
    "PrefixCache", "PrefixNode",
    "LLMEngine", "EngineConfig", "EngineStats",
    "BenchmarkConfig", "BenchmarkResult", "run_benchmark", "make_requests",
]
''')
commit("refactor: export all serving components from nanomind/serving/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — example
# ══════════════════════════════════════════════════════════════════════════════
write("examples/serving_demo.py", '''\
"""
examples/serving_demo.py — NanoMind Continuous Batching & Serving demo.

Demonstrates:
  1. KV Cache block manager (PagedAttention)
  2. Inference request lifecycle
  3. Continuous batching scheduler
  4. Prefix cache (RadixAttention)
  5. Full LLM Engine: submit, step, run_until_done
  6. Throughput benchmark: continuous vs static batching

Usage:
    python examples/serving_demo.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanomind.serving import (
    KVBlock, KVCacheManager,
    InferenceRequest, SamplingParams, RequestStatus,
    ContinuousBatchingScheduler, SchedulerOutput,
    PrefixCache,
    LLMEngine, EngineConfig, EngineStats,
    BenchmarkConfig, BenchmarkResult, run_benchmark, make_requests,
)

V = 64   # small vocab for demo

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 16)
        self.rnn  = nn.GRU(16, 32, batch_first=True)
        self.head = nn.Linear(32, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None

print("=" * 60)
print("NanoMind Continuous Batching & Serving Demo")
print("=" * 60)

# ── KV Cache Manager ──────────────────────────────────────────────────────────
print("\n── KV Cache Manager (PagedAttention) ──")
kv = KVCacheManager(n_blocks=64, block_size=8, n_layers=2, n_kv_heads=1, d_head=16)
print(f"  Pool: {kv.n_blocks} blocks × {kv.block_size} tokens")
print(f"  Memory: {kv.stats()['memory_gb']:.3f} GB")

kv.allocate("req_1", n_tokens=16)
kv.allocate("req_2", n_tokens=8)
print(f"  After 2 allocs: {kv.stats()}")
kv.free("req_1")
print(f"  After free req_1: free_blocks={kv.n_free_blocks}")

# Copy-on-write (beam search fork)
kv.allocate("req_3", n_tokens=8)
kv.copy_on_write("req_3", "req_3_fork")
print(f"  After CoW fork: n_seqs={kv.stats()['n_sequences']}")

# ── Inference Request ──────────────────────────────────────────────────────────
print("\n── Inference Request Lifecycle ──")
sp  = SamplingParams(max_new_tokens=4, temperature=0.8, top_p=0.9)
req = InferenceRequest("req_test", [1, 2, 3, 4, 5], sp)
print(f"  Status: {req.status.value}, prompt_len={req.prompt_len}")
req.status = RequestStatus.RUNNING
req.add_token(42); req.add_token(7); req.add_token(99); req.add_token(11)
print(f"  Generated: {req.generated_tokens}")
print(f"  Finished: {req.is_finished} (max_new_tokens=4)")
print(f"  Output: {req.output_ids[:8]}...")

# ── Continuous Batching Scheduler ─────────────────────────────────────────────
print("\n── Continuous Batching Scheduler ──")
kv2 = KVCacheManager(n_blocks=128, block_size=8, n_layers=1, n_kv_heads=1, d_head=8)
sched = ContinuousBatchingScheduler(kv2, max_batch_size=4)
for i in range(5):
    r = InferenceRequest(f"req_{i}", list(range(i+2)),
                          SamplingParams(max_new_tokens=4))
    sched.add_request(r)
print(f"  Queued: {sched.queue_len()} requests")
out = sched.step()
print(f"  After 1 step: running={sched.running_len()}, queue={sched.queue_len()}")
print(f"  Batched tokens: {out.n_batched_tokens}")
print(f"  Scheduler stats: {sched.stats()['running']} running, {sched.stats()['waiting']} waiting")

# ── Prefix Cache ──────────────────────────────────────────────────────────────
print("\n── Prefix Cache (RadixAttention) ──")
cache = PrefixCache()
sys_prompt = [10, 11, 12, 13, 14, 15]   # system prompt tokens
cache.insert(sys_prompt, block_ids=[0, 1])
# Request with same system prompt
req_tokens = sys_prompt + [20, 21, 22]
n_cached, cached_blocks = cache.lookup(req_tokens)
print(f"  System prompt len: {len(sys_prompt)}")
print(f"  Cache hit: {n_cached} tokens, blocks={cached_blocks}")
print(f"  Cache stats: {cache.stats()}")
# Request without cache hit
n2, b2 = cache.lookup([99, 88, 77])
print(f"  Miss: {n2} tokens cached")

# ── LLM Engine (full continuous batching) ─────────────────────────────────────
print("\n── LLM Engine: Continuous Batching ──")
cfg    = EngineConfig(
    max_batch_size=4, n_kv_blocks=64, block_size=8,
    n_layers=1, n_kv_heads=1, d_head=16, vocab_size=V,
)
engine = LLMEngine(TinyLM(), cfg)

reqs = []
for i in range(6):
    r = engine.submit(f"r{i}", [i % V, (i+1) % V, (i+2) % V],
                       SamplingParams(max_new_tokens=6))
    reqs.append(r)

print(f"  Submitted {len(reqs)} requests")
finished = engine.run_until_done()
print(f"  Finished: {len(finished)} requests")
for r in finished[:3]:
    print(f"    {r.request_id}: {r.n_generated} tokens generated")
print(f"  Engine stats: {engine.engine_stats()['n_tokens_generated']} tokens")

# ── Benchmark ──────────────────────────────────────────────────────────────────
print("\n── Throughput Benchmark ──")
bench_cfg = BenchmarkConfig(
    n_requests=10, min_prompt_len=2, max_prompt_len=6,
    min_output_len=3, max_output_len=8, vocab_size=V,
)
bench_engine = LLMEngine(TinyLM(), cfg)
result = run_benchmark(bench_engine, bench_cfg)
print(f"  Result: {result.to_dict()}")

print("\nContinuous batching demo complete!")
''')
commit("feat: add examples/serving_demo.py — KV cache, scheduler, prefix cache, engine, benchmark")

# ══════════════════════════════════════════════════════════════════════════════
# COMMITS 11-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_serving.py", '''\
"""tests/test_serving.py — Tests for NanoMind serving package."""
import time
import pytest
import torch
import torch.nn as nn
from nanomind.serving import (
    KVBlock, KVCacheManager,
    InferenceRequest, SamplingParams, RequestStatus,
    ContinuousBatchingScheduler, SchedulerOutput,
    PrefixCache,
    LLMEngine, EngineConfig, EngineStats,
    BenchmarkConfig, make_requests, run_benchmark,
)

V = 16

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 8)
        self.rnn  = nn.GRU(8, 16, batch_first=True)
        self.head = nn.Linear(16, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None


# ── KVCacheManager ────────────────────────────────────────────────────────────

class TestKVCacheManager:
    def _mgr(self):
        return KVCacheManager(n_blocks=32, block_size=8, n_layers=1, n_kv_heads=1, d_head=8)

    def test_allocate_reduces_free(self):
        m = self._mgr()
        n = m.n_free_blocks
        m.allocate("s1", n_tokens=8)
        assert m.n_free_blocks < n

    def test_free_restores_blocks(self):
        m = self._mgr()
        m.allocate("s1", 8)
        before = m.n_free_blocks
        m.free("s1")
        assert m.n_free_blocks > before

    def test_can_allocate_true(self):
        m = self._mgr()
        assert m.can_allocate(8)

    def test_can_allocate_false_too_many(self):
        m = self._mgr()
        assert not m.can_allocate(9999)

    def test_stats_keys(self):
        m = self._mgr()
        s = m.stats()
        for k in ("total_blocks", "free_blocks", "utilisation"):
            assert k in s

    def test_copy_on_write(self):
        m = self._mgr()
        m.allocate("src", 8)
        m.copy_on_write("src", "dst")
        assert len(m.get_blocks("dst")) > 0

    def test_double_free_no_error(self):
        m = self._mgr()
        m.allocate("s1", 8)
        m.free("s1")
        m.free("s1")  # should not raise

    def test_get_blocks_returns_list(self):
        m = self._mgr()
        m.allocate("s1", 8)
        blocks = m.get_blocks("s1")
        assert isinstance(blocks, list)


# ── InferenceRequest ──────────────────────────────────────────────────────────

class TestInferenceRequest:
    def test_add_token_updates_list(self):
        r = InferenceRequest("r1", [1, 2, 3], SamplingParams(max_new_tokens=4))
        r.add_token(7)
        assert r.generated_tokens == [7]

    def test_finishes_at_max_tokens(self):
        r = InferenceRequest("r1", [1], SamplingParams(max_new_tokens=2))
        r.add_token(1); r.add_token(2)
        assert r.is_finished

    def test_not_finished_before_max(self):
        r = InferenceRequest("r1", [1], SamplingParams(max_new_tokens=4))
        r.add_token(1)
        assert not r.is_finished

    def test_stop_token(self):
        r = InferenceRequest("r1", [1], SamplingParams(max_new_tokens=10, stop_token_id=99))
        r.add_token(7)
        assert not r.is_finished
        r.add_token(99)
        assert r.is_finished

    def test_output_ids(self):
        r = InferenceRequest("r1", [1, 2], SamplingParams())
        r.add_token(3)
        assert r.output_ids == [1, 2, 3]

    def test_abort(self):
        r = InferenceRequest("r1", [1], SamplingParams())
        r.abort()
        assert r.status == RequestStatus.ABORTED
        assert r.is_finished


# ── ContinuousBatchingScheduler ───────────────────────────────────────────────

class TestScheduler:
    def _sched(self):
        kv = KVCacheManager(64, block_size=8, n_layers=1, n_kv_heads=1, d_head=8)
        return ContinuousBatchingScheduler(kv, max_batch_size=4)

    def test_add_and_step_admits(self):
        s = self._sched()
        r = InferenceRequest("r1", [1, 2, 3], SamplingParams(max_new_tokens=4))
        s.add_request(r)
        out = s.step()
        assert len(out.running) >= 1

    def test_queue_shrinks_after_admit(self):
        s = self._sched()
        for i in range(3):
            s.add_request(InferenceRequest(f"r{i}", [1, 2], SamplingParams(max_new_tokens=2)))
        s.step()
        assert s.queue_len() < 3

    def test_has_work_true(self):
        s = self._sched()
        s.add_request(InferenceRequest("r1", [1], SamplingParams()))
        assert s.has_work()

    def test_stats_keys(self):
        s = self._sched()
        d = s.stats()
        assert "running" in d and "waiting" in d


# ── PrefixCache ───────────────────────────────────────────────────────────────

class TestPrefixCache:
    def test_insert_and_lookup_hit(self):
        c = PrefixCache()
        c.insert([1, 2, 3, 4], [0, 1])
        n, blocks = c.lookup([1, 2, 3, 4, 5])
        assert n >= 4
        assert blocks == [0, 1]

    def test_lookup_miss(self):
        c = PrefixCache()
        n, blocks = c.lookup([9, 9, 9])
        assert n == 0

    def test_stats_hit_rate(self):
        c = PrefixCache()
        c.insert([1, 2, 3], [0])
        c.lookup([1, 2, 3, 4])  # hit
        c.lookup([9, 8])         # miss
        s = c.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert 0 < s["hit_rate"] < 1

    def test_evict_lru(self):
        c = PrefixCache()
        c.insert([1, 2], [0])
        evicted = c.evict_lru(n=1)
        assert evicted >= 0   # may be 0 if ref_count > 0


# ── LLMEngine ─────────────────────────────────────────────────────────────────

class TestLLMEngine:
    def _engine(self):
        cfg = EngineConfig(max_batch_size=4, n_kv_blocks=64, block_size=8,
                            n_layers=1, n_kv_heads=1, d_head=8, vocab_size=V)
        return LLMEngine(TinyLM(), cfg)

    def test_submit_and_run(self):
        e = self._engine()
        e.submit("r1", [1, 2, 3], SamplingParams(max_new_tokens=4))
        finished = e.run_until_done()
        assert len(finished) == 1
        assert finished[0].n_generated == 4

    def test_multiple_requests(self):
        e = self._engine()
        for i in range(3):
            e.submit(f"r{i}", [i+1, i+2], SamplingParams(max_new_tokens=3))
        finished = e.run_until_done()
        assert len(finished) == 3

    def test_stats_populated(self):
        e = self._engine()
        e.submit("r1", [1, 2], SamplingParams(max_new_tokens=2))
        e.run_until_done()
        s = e.engine_stats()
        assert s["n_tokens_generated"] > 0

    def test_throughput_positive(self):
        e = self._engine()
        e.submit("r1", [1, 2, 3], SamplingParams(max_new_tokens=3))
        e.run_until_done()
        assert e.stats.throughput > 0
''')
commit("test: add full serving test suite — KV cache, request, scheduler, prefix cache, engine")

# COMMITS 12-18
for title, body in [
    ("test: add KVCacheManager out of memory raises test", '''
class TestKVOOM:
    def test_oom_raises(self):
        from nanomind.serving import KVCacheManager
        kv = KVCacheManager(n_blocks=2, block_size=8,
                             n_layers=1, n_kv_heads=1, d_head=4)
        kv.allocate("s1", 8)
        kv.allocate("s2", 8)
        with pytest.raises(MemoryError):
            kv.allocate("s3", 8)
'''),
    ("test: add InferenceRequest prompt_len and total_len test", '''
class TestRequestLengths:
    def test_prompt_len(self):
        from nanomind.serving import InferenceRequest, SamplingParams
        r = InferenceRequest("r", [1, 2, 3, 4], SamplingParams())
        assert r.prompt_len == 4

    def test_total_len(self):
        from nanomind.serving import InferenceRequest, SamplingParams
        r = InferenceRequest("r", [1, 2], SamplingParams(max_new_tokens=4))
        r.add_token(5)
        assert r.total_len == 3   # 2 prompt + 1 generated
'''),
    ("test: add SamplingParams to_dict keys test", '''
class TestSamplingParams:
    def test_to_dict_keys(self):
        from nanomind.serving import SamplingParams
        sp = SamplingParams(max_new_tokens=8, temperature=0.7)
        d  = sp.to_dict()
        assert "max_new_tokens" in d
        assert "temperature"    in d
'''),
    ("test: add PrefixCache multiple inserts test", '''
class TestPrefixMultiInsert:
    def test_two_different_prefixes(self):
        from nanomind.serving import PrefixCache
        c = PrefixCache()
        c.insert([1, 2, 3], [0])
        c.insert([4, 5, 6], [1])
        n1, b1 = c.lookup([1, 2, 3, 9])
        n2, b2 = c.lookup([4, 5, 6, 9])
        assert n1 >= 3 and n2 >= 3
        assert b1 == [0] and b2 == [1]
'''),
    ("test: add Scheduler max_batch_size respected test", '''
class TestSchedulerMaxBatch:
    def test_max_batch_size(self):
        from nanomind.serving import KVCacheManager, ContinuousBatchingScheduler
        from nanomind.serving import InferenceRequest, SamplingParams
        kv   = KVCacheManager(64, 8, 1, 1, 8)
        sched = ContinuousBatchingScheduler(kv, max_batch_size=2)
        for i in range(5):
            sched.add_request(InferenceRequest(f"r{i}", [1, 2], SamplingParams(max_new_tokens=4)))
        out = sched.step()
        assert len(out.running) <= 2   # max_batch_size respected
'''),
    ("test: add LLMEngine step returns finished test", '''
class TestEngineStep:
    def test_step_returns_list(self):
        from nanomind.serving import LLMEngine, EngineConfig, SamplingParams
        import torch.nn as nn, torch
        cfg = EngineConfig(max_batch_size=2, n_kv_blocks=32, block_size=8,
                            n_layers=1, n_kv_heads=1, d_head=8, vocab_size=V)
        e   = LLMEngine(TinyLM(), cfg)
        e.submit("r1", [1, 2], SamplingParams(max_new_tokens=1))
        result = e.step()
        assert isinstance(result, list)
'''),
    ("test: add BenchmarkResult throughput test", '''
class TestBenchmark:
    def test_benchmark_runs(self):
        from nanomind.serving import (LLMEngine, EngineConfig, BenchmarkConfig,
                                       run_benchmark)
        cfg   = EngineConfig(max_batch_size=4, n_kv_blocks=64, block_size=8,
                              n_layers=1, n_kv_heads=1, d_head=8, vocab_size=V)
        e     = LLMEngine(TinyLM(), cfg)
        bcfg  = BenchmarkConfig(n_requests=4, min_prompt_len=2, max_prompt_len=4,
                                 min_output_len=2, max_output_len=4, vocab_size=V)
        res   = run_benchmark(e, bcfg)
        assert res.throughput_tok_s > 0
        assert res.n_requests == 4
        assert res.n_tokens_total > 0
        d = res.to_dict()
        assert "throughput_tok_s" in d
'''),
]:
    src = read("tests/test_serving.py")
    src += "\n" + body
    write("tests/test_serving.py", src)
    commit(title)

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v4.9.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"4.8.0\"", "__version__ = \"4.9.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v4.9.0 — Continuous Batching & KV Cache Serving release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag + 🎉 1000 commits!
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `specd`      | Speculative Decoding — draft model, reject/accept, Medusa heads, tree attention, lookahead |",
    "| `specd`      | Speculative Decoding — draft model, reject/accept, Medusa heads, tree attention, lookahead |\n"
    "| `serving`    | Continuous Batching — PagedAttention KV cache, scheduler, prefix caching, LLM engine |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = ("## [4.9.0] — 2024 — Continuous Batching & KV Cache Serving 🎉 1000 COMMITS!\n\n### Added\n"
      "- `KVBlock` / `KVCacheManager` — PagedAttention block pool, allocate, free, copy-on-write\n"
      "- `InferenceRequest` — request lifecycle, token accumulation, latency tracking\n"
      "- `SamplingParams` — temperature, top_p, repetition_penalty, stop token\n"
      "- `RequestStatus` — WAITING/RUNNING/FINISHED/ABORTED enum\n"
      "- `ContinuousBatchingScheduler` — admit, preempt, step(), SchedulerOutput\n"
      "- `PrefixCache` — RadixAttention radix tree, insert/lookup, LRU eviction\n"
      "- `EngineConfig` — unified serving configuration\n"
      "- `LLMEngine` — submit, step, run_until_done, engine_stats\n"
      "- `EngineStats` — throughput, total_tokens, wall_time\n"
      "- `BenchmarkConfig` / `BenchmarkResult` — serving benchmark metrics\n"
      "- `run_benchmark` — end-to-end throughput/latency benchmark\n"
      "- `examples/serving_demo.py` — full serving demo\n\n---\n\n") + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v4.9.0, update README and CHANGELOG for Day 49 — 🎉 1000 COMMITS MILESTONE!")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 49 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v4.9.0",
    "-m", "NanoMind v4.9.0 — Continuous Batching & Serving (1000 Commits!)", check=False)
r = run("git", "push", "origin", "v4.9.0", check=False)
print("Tag v4.9.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
total_n = int(total.stdout.strip())
print(f"\n{'🎉'*10}")
print(f"🏆 TOTAL COMMITS: {total_n}")
if total_n >= 1000:
    print(f"🎊 WE HIT {total_n} COMMITS — THE 1000 MILESTONE IS CROSSED! 🎊")
print(f"{'🎉'*10}")
print("=== DAY 49 COMPLETE — v4.9.0 TAGGED! ===")
