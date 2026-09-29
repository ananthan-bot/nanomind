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
