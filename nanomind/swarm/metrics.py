"""
nanomind/swarm/metrics.py — Multi-agent swarm evaluation metrics: Consensus Rate, Communication Efficiency.
"""
from typing import List, Dict, Any


def compute_consensus_agreement_rate(votes_list: List[List[str]]) -> float:
    """
    Computes average agreement rate across multiple consensus decisions.
    """
    if not votes_list:
        return 0.0

    agreements = []
    for votes in votes_list:
        if not votes:
            continue
        from collections import Counter
        counts = Counter(votes)
        top = counts.most_common(1)[0][1]
        agreements.append(top / len(votes))

    return sum(agreements) / max(1, len(agreements))


def compute_communication_overhead(message_history: List[Any], num_tasks: int = 1) -> Dict[str, float]:
    """
    Measures message exchange density per completed task.
    """
    total_messages = len(message_history)
    return {
        "total_messages": total_messages,
        "messages_per_task": round(total_messages / max(1, num_tasks), 2),
    }
