"""
nanomind/specd/tree.py — Token tree for candidate exploration.

In standard speculative decoding: K tokens in a linear sequence.
In tree speculative decoding: a TREE of candidate token sequences.

Example K=3, top_k=2 per head:
  Root (current context)
    ├─ Token A (prob 0.4)
    │   ├─ Token C (prob 0.3)
    │   └─ Token D (prob 0.2)
    └─ Token B (prob 0.3)
        └─ Token E (prob 0.4)

Tree attention: attend to all paths simultaneously.
Accept the longest valid prefix from the tree.

This is a simplified version of SpecTr (Sun et al., 2023)
and Medusa's tree decoding.
"""

from __future__ import annotations
import torch
from dataclasses import dataclass, field


@dataclass
class TreeNode:
    """A node in the candidate token tree."""
    token_id:  int
    prob:      float
    depth:     int
    parent:    "TreeNode | None"  = None
    children:  list               = field(default_factory=list)
    node_id:   int                = 0

    def path_to_root(self) -> list[int]:
        """Get token sequence from root to this node."""
        path = []
        node = self
        while node.parent is not None:
            path.append(node.token_id)
            node = node.parent
        return path[::-1]

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class TokenTree:
    """
    A tree of candidate token sequences for speculative decoding.

    Each path from root to leaf represents a candidate continuation.

    Args:
        max_depth: Maximum tree depth (= speculation length K).
        branching: Number of candidates per node (top-K per head).

    Example::

        tree   = TokenTree(max_depth=3, branching=2)
        root   = tree.build(candidate_tokens)
        paths  = tree.all_paths()
    """

    def __init__(self, max_depth: int = 4, branching: int = 2) -> None:
        self.max_depth = max_depth
        self.branching = branching
        self._node_counter = 0

    def _new_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def build(
        self,
        candidate_tokens: list[list[int]],
        candidate_probs:  list[list[float]] | None = None,
    ) -> TreeNode:
        """
        Build tree from per-depth candidate lists.

        Args:
            candidate_tokens: List of K lists, each with branching token IDs.
                              candidate_tokens[d] = [tok_1, tok_2, ...] at depth d.
            candidate_probs:  Optional probabilities.

        Returns:
            Root :class:`TreeNode`.
        """
        root = TreeNode(token_id=-1, prob=1.0, depth=0, node_id=0)
        if not candidate_tokens:
            return root

        probs = candidate_probs or [[1.0] * len(t) for t in candidate_tokens]
        self._node_counter = 0

        # Build level by level
        leaves = [root]
        for depth, (tokens, plist) in enumerate(zip(candidate_tokens, probs)):
            new_leaves = []
            for parent in leaves:
                for tok, p in zip(tokens[:self.branching], plist[:self.branching]):
                    child = TreeNode(
                        token_id = tok,
                        prob     = p,
                        depth    = depth + 1,
                        parent   = parent,
                        node_id  = self._new_id(),
                    )
                    parent.children.append(child)
                    new_leaves.append(child)
                if depth >= self.max_depth - 1:
                    break   # don't expand further
            leaves = new_leaves
        return root

    def all_paths(self, root: TreeNode) -> list[list[int]]:
        """Return all root-to-leaf paths as token sequences."""
        paths = []
        def dfs(node, path):
            if node.is_leaf():
                if path:
                    paths.append(path[:])
                return
            for child in node.children:
                dfs(child, path + [child.token_id])
        dfs(root, [])
        return paths

    def n_nodes(self, root: TreeNode) -> int:
        """Count total nodes in tree."""
        count = [0]
        def dfs(n):
            count[0] += 1
            for c in n.children:
                dfs(c)
        dfs(root)
        return count[0]

    def verify_paths(
        self,
        root:          TreeNode,
        target_tokens: list[int],
    ) -> list[int]:
        """
        Find longest path that matches target_tokens prefix.

        Args:
            root:          Tree root.
            target_tokens: Target model's predicted tokens (one per position).

        Returns:
            Best matching path (list of token IDs).
        """
        best  = []
        def dfs(node, path, depth):
            nonlocal best
            if depth >= len(target_tokens):
                if len(path) > len(best):
                    best = path[:]
                return
            if node.token_id != target_tokens[depth - 1] and depth > 0:
                if len(path) > len(best):
                    best = path[:]
                return
            for child in node.children:
                dfs(child, path + [child.token_id], depth + 1)
        dfs(root, [], 0)
        return best
