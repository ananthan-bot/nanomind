"""
nanomind.swarm — Autonomous Multi-Agent Swarms & Compound AI Systems.
"""
from nanomind.swarm.config import (
    SwarmTopology,
    AgentRole,
    SwarmConfig,
)
from nanomind.swarm.message import (
    MessageEnvelope,
    MessageBroker,
)
from nanomind.swarm.blackboard import (
    Blackboard,
)
from nanomind.swarm.agent import (
    SwarmAgent,
)
from nanomind.swarm.handoff import (
    HandoffContext,
    HandoffProtocol,
)
from nanomind.swarm.consensus import (
    ConsensusEngine,
)
from nanomind.swarm.debate import (
    MultiAgentDebate,
)
from nanomind.swarm.specialists import (
    ArchitectAgent,
    CoderAgent,
    ReviewerAgent,
)
from nanomind.swarm.orchestrator import (
    SwarmOrchestrator,
)
from nanomind.swarm.metrics import (
    compute_consensus_agreement_rate,
    compute_communication_overhead,
)

__all__ = [
    "SwarmTopology",
    "AgentRole",
    "SwarmConfig",
    "MessageEnvelope",
    "MessageBroker",
    "Blackboard",
    "SwarmAgent",
    "HandoffContext",
    "HandoffProtocol",
    "ConsensusEngine",
    "MultiAgentDebate",
    "ArchitectAgent",
    "CoderAgent",
    "ReviewerAgent",
    "SwarmOrchestrator",
    "compute_consensus_agreement_rate",
    "compute_communication_overhead",
]
