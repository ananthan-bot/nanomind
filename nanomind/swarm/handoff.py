"""
nanomind/swarm/handoff.py — OpenAI Swarm style dynamic agent handoff protocol.
Enables agents to transfer execution context and control to another specialized agent.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class HandoffContext:
    """Execution state transferred during handoff."""
    source_agent_id: str
    target_agent_id: str
    reason: str
    task_state: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class HandoffProtocol:
    """
    Manages structured transfers of control between agents.
    Verifies target availability and packages context state.
    """

    def __init__(self):
        self.handoff_log: List[HandoffContext] = []

    def execute_handoff(
        self,
        source_agent_id: str,
        target_agent_id: str,
        reason: str,
        task_state: Optional[Dict[str, Any]] = None,
    ) -> HandoffContext:
        ctx = HandoffContext(
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            reason=reason,
            task_state=task_state or {},
        )
        self.handoff_log.append(ctx)
        return ctx

    def get_history(self) -> List[HandoffContext]:
        return list(self.handoff_log)
