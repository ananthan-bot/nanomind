"""
nanomind/swarm/config.py — Configuration dataclasses, agent roles, and swarm network topologies.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


class SwarmTopology(str, Enum):
    """Network communication topology for agent swarms."""
    HIERARCHICAL = "hierarchical"  # Leader orchestrator delegates to worker specialists
    PEER_TO_PEER = "peer_to_peer"  # Mesh network: all agents broadcast and interact directly
    CIRCULAR = "circular"          # Ring pipeline: Agent A -> Agent B -> Agent C -> Agent A
    DEBATE = "debate"              # Competing agent pairs moderated by an arbiter


class AgentRole(str, Enum):
    """Predefined functional roles for specialized swarm agents."""
    ORCHESTRATOR = "orchestrator"
    ARCHITECT = "architect"
    CODER = "coder"
    REVIEWER = "reviewer"
    AUDITOR = "auditor"
    VERIFIER = "verifier"


@dataclass
class SwarmConfig:
    """Master configuration for multi-agent swarm system."""
    topology: SwarmTopology = SwarmTopology.HIERARCHICAL
    max_rounds: int = 8
    consensus_threshold: float = 0.67  # 2/3 supermajority for critical decisions
    timeout_seconds: float = 30.0
    enable_blackboard: bool = True
    max_history_length: int = 50
    default_roles: List[AgentRole] = field(
        default_factory=lambda: [AgentRole.ORCHESTRATOR, AgentRole.CODER, AgentRole.REVIEWER]
    )
