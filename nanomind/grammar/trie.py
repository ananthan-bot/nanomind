"""
nanomind/grammar/trie.py — Prefix Trie for fast vocabulary token retrieval and prefix matching.
"""
from typing import Dict, List, Set, Optional, Tuple


class TrieNode:
    """A node in the prefix trie representing a character transition."""
    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.token_ids: List[int] = []
        self.is_leaf: bool = False


class PrefixTrie:
    """
    Prefix trie constructed over a vocabulary of tokens to enable O(L) prefix querying
    and grammar constraint masking.
    """
    def __init__(self):
        self.root = TrieNode()
        self._vocab_size = 0

    def insert(self, token_str: str, token_id: int) -> None:
        """Insert a vocabulary token into the trie."""
        curr = self.root
        for ch in token_str:
            if ch not in curr.children:
                curr.children[ch] = TrieNode()
            curr = curr.children[ch]
        curr.token_ids.append(token_id)
        curr.is_leaf = True
        self._vocab_size += 1

    def find_tokens_with_prefix(self, prefix: str) -> List[int]:
        """Retrieve all token IDs that begin with the specified prefix."""
        curr = self.root
        for ch in prefix:
            if ch not in curr.children:
                return []
            curr = curr.children[ch]
        return self._collect_all_tokens(curr)

    def _collect_all_tokens(self, node: TrieNode) -> List[int]:
        tokens = list(node.token_ids)
        for child in node.children.values():
            tokens.extend(self._collect_all_tokens(child))
        return tokens

    @classmethod
    def from_vocab(cls, vocab: Dict[int, str]) -> "PrefixTrie":
        """Factory method building a trie from a token_id -> string vocabulary dict."""
        trie = cls()
        for tid, s in vocab.items():
            trie.insert(s, tid)
        return trie
