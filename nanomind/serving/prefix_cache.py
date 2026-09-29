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
