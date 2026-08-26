"""
nanomind/speculative/tracker.py — Track speculative decoding statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpeculativeStats:
    """
    Running statistics for a speculative decoding session.

    Attributes:
        total_tokens:   Total tokens generated.
        total_draft:    Total draft tokens proposed.
        total_accepted: Total draft tokens accepted.
        n_calls:        Number of speculative decoding calls.
    """

    total_tokens:   int = 0
    total_draft:    int = 0
    total_accepted: int = 0
    n_calls:        int = 0

    def update(self, stats: dict) -> None:
        """Accumulate stats from one speculative_decode() call."""
        self.total_tokens   += stats.get("n_tokens", 0)
        self.total_draft    += stats.get("total_draft", 0)
        self.total_accepted += stats.get("total_accepted", 0)
        self.n_calls        += 1

    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate across all calls."""
        return self.total_accepted / max(self.total_draft, 1)

    @property
    def mean_tokens_per_call(self) -> float:
        """Average number of tokens generated per speculative call."""
        return self.total_tokens / max(self.n_calls, 1)

    def __str__(self) -> str:
        return (
            f"SpeculativeStats("
            f"tokens={self.total_tokens}, "
            f"acceptance={self.acceptance_rate:.2%}, "
            f"calls={self.n_calls})"
        )


def print_speculative_report(stats: dict) -> None:
    """Pretty-print a single speculative_decode() stats dict."""
    print("=" * 50)
    print("Speculative Decoding Report")
    print("=" * 50)
    print(f"  Tokens generated : {stats.get('n_tokens', 0):>6}")
    print(f"  Draft calls      : {stats.get('n_draft_calls', 0):>6}")
    print(f"  Draft tokens     : {stats.get('total_draft', 0):>6}")
    print(f"  Accepted tokens  : {stats.get('total_accepted', 0):>6}")
    ar = stats.get("acceptance_rate", 0)
    print(f"  Acceptance rate  : {ar:>6.2%}")
    print("=" * 50)
