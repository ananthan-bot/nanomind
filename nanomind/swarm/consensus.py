"""
nanomind/swarm/consensus.py — Consensus voting mechanisms: Majority, Supermajority, Weighted, and Borda Count.
"""
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter


class ConsensusEngine:
    """
    Coordinates consensus across agents when verifying solutions or approving actions.
    """

    @staticmethod
    def majority_vote(votes: List[str]) -> Tuple[Optional[str], float]:
        """
        Standard majority voting.
        Returns: (winning_choice, agreement_fraction)
        """
        if not votes:
            return None, 0.0
        counts = Counter(votes)
        winner, top_count = counts.most_common(1)[0]
        fraction = top_count / len(votes)
        return winner, float(fraction)

    @staticmethod
    def weighted_consensus(votes_with_weights: List[Tuple[str, float]]) -> Tuple[Optional[str], float]:
        """
        Weighted consensus using agent confidence or reputation weights.
        votes_with_weights: [(choice, weight), ...]
        """
        if not votes_with_weights:
            return None, 0.0

        tally: Dict[str, float] = {}
        total_weight = 0.0

        for choice, weight in votes_with_weights:
            tally[choice] = tally.get(choice, 0.0) + weight
            total_weight += weight

        if total_weight == 0.0:
            return None, 0.0

        winner = max(tally.items(), key=lambda x: x[1])
        return winner[0], float(winner[1] / total_weight)

    @staticmethod
    def borda_count(rankings: List[List[str]]) -> List[Tuple[str, int]]:
        """
        Borda count preferential voting:
        For N candidates, rank 1 gets N points, rank 2 gets N-1, etc.
        rankings: list of agent preference rankings, e.g. [["optA", "optB", "optC"], ...]
        """
        if not rankings:
            return []

        points: Dict[str, int] = {}
        for ranking in rankings:
            n = len(ranking)
            for idx, item in enumerate(ranking):
                pts = n - idx
                points[item] = points.get(item, 0) + pts

        sorted_results = sorted(points.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
