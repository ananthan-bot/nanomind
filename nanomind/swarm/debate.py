"""
nanomind/swarm/debate.py — Multi-Agent Debate for hallucination reduction and solution verification.
(Du et al., 2023: Improving Factuality and Reasoning through Multiagent Debate)
"""
from typing import List, Dict, Any, Tuple, Optional, Callable


class MultiAgentDebate:
    """
    Orchestrates round-based adversarial or collaborative debate between multiple agents.
    Exchanges arguments, counter-arguments, and synthesizes consensus through an arbiter.
    """

    def __init__(self, debater_ids: List[str], max_rounds: int = 3):
        self.debater_ids = debater_ids
        self.max_rounds = max_rounds

    def run_debate(
        self,
        topic: str,
        agent_respond_fn: Callable[[str, str, List[Dict[str, str]]], str],
        arbiter_judge_fn: Callable[[str, List[Dict[str, str]]], str],
    ) -> Dict[str, Any]:
        """
        Runs multi-round debate.
        agent_respond_fn(agent_id, topic, round_history) -> argument_text
        arbiter_judge_fn(topic, full_debate_transcript) -> final_decision
        """
        transcript: List[Dict[str, str]] = []

        for r in range(1, self.max_rounds + 1):
            round_arguments = {}
            for agent_id in self.debater_ids:
                argument = agent_respond_fn(agent_id, topic, transcript)
                round_arguments[agent_id] = argument
                transcript.append({
                    "round": str(r),
                    "agent": agent_id,
                    "argument": argument,
                })

        final_verdict = arbiter_judge_fn(topic, transcript)

        return {
            "topic": topic,
            "rounds_completed": self.max_rounds,
            "transcript": transcript,
            "final_verdict": final_verdict,
        }
