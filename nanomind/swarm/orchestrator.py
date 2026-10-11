"""
nanomind/swarm/orchestrator.py — Master SwarmOrchestrator executing compound AI pipelines.
"""
from typing import Dict, Any, List, Optional
from nanomind.swarm.config import SwarmConfig, SwarmTopology
from nanomind.swarm.message import MessageBroker
from nanomind.swarm.blackboard import Blackboard
from nanomind.swarm.specialists import ArchitectAgent, CoderAgent, ReviewerAgent
from nanomind.swarm.handoff import HandoffProtocol
from nanomind.swarm.consensus import ConsensusEngine


class SwarmOrchestrator:
    """
    Coordinates multi-agent workflows, dispatches tasks, monitors blackboard updates,
    and conducts consensus verification across agents.
    """

    def __init__(self, config: Optional[SwarmConfig] = None):
        self.config = config or SwarmConfig()
        self.broker = MessageBroker()
        self.blackboard = Blackboard()
        self.handoff_protocol = HandoffProtocol()

        # Initialize core swarm specialists
        self.architect = ArchitectAgent(broker=self.broker, blackboard=self.blackboard)
        self.coder = CoderAgent(broker=self.broker, blackboard=self.blackboard)
        self.reviewer = ReviewerAgent(broker=self.broker, blackboard=self.blackboard)

    def execute_compound_pipeline(self, user_prompt: str) -> Dict[str, Any]:
        """
        Executes an end-to-end multi-agent compound task:
        1. Architect plans task
        2. Handoff to Coder for implementation
        3. Handoff to Reviewer for verification
        4. Consensus approval
        """
        # Step 1: Architect planning
        plan = self.architect.plan_architecture(user_prompt)

        # Step 2: Handoff to Coder
        self.handoff_protocol.execute_handoff(
            source_agent_id="architect",
            target_agent_id="coder",
            reason="Delegate implementation to Coder",
            task_state=plan,
        )
        artifact = self.coder.implement(plan)

        # Step 3: Handoff to Reviewer
        self.handoff_protocol.execute_handoff(
            source_agent_id="coder",
            target_agent_id="reviewer",
            reason="Delegate code review to Reviewer",
            task_state=artifact,
        )
        review = self.reviewer.review(artifact)

        # Step 4: Consensus Check
        votes = ["approve" if review["approved"] else "reject", "approve"]  # Architect + Reviewer votes
        winner, agreement = ConsensusEngine.majority_vote(votes)

        return {
            "user_prompt": user_prompt,
            "plan": plan,
            "artifact": artifact,
            "review": review,
            "consensus_verdict": winner,
            "agreement_rate": agreement,
            "handoffs_count": len(self.handoff_protocol.get_history()),
            "blackboard_snapshot": self.blackboard.snapshot(),
        }
