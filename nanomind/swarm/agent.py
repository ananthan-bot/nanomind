"""
nanomind/swarm/agent.py — SwarmAgent base class with state lifecycle, local memory, and message handling.
"""
from typing import List, Dict, Any, Optional
from nanomind.swarm.config import AgentRole
from nanomind.swarm.message import MessageEnvelope, MessageBroker
from nanomind.swarm.blackboard import Blackboard


class SwarmAgent:
    """
    Base autonomous agent in the swarm.
    Possesses local memory, assigned role, communication hooks, and tool execution.
    """

    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        broker: Optional[MessageBroker] = None,
        blackboard: Optional[Blackboard] = None,
    ):
        self.agent_id = agent_id
        self.role = role
        self.broker = broker
        self.blackboard = blackboard
        self.local_memory: List[Dict[str, Any]] = []

        if self.broker:
            self.broker.register_agent(self.agent_id)

    def send_message(self, recipient_id: str, topic: str, payload: Any):
        """Send message via broker."""
        if not self.broker:
            return
        msg = MessageEnvelope(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            topic=topic,
            payload=payload,
        )
        self.broker.publish(msg)

    def receive_messages(self) -> List[MessageEnvelope]:
        """Fetch unread messages from broker inbox."""
        if not self.broker:
            return []
        return self.broker.pop_inbox(self.agent_id)

    def write_blackboard(self, key: str, value: Any):
        if self.blackboard:
            self.blackboard.write(key, value, self.agent_id)

    def read_blackboard(self, key: str, default: Any = None) -> Any:
        if self.blackboard:
            return self.blackboard.read(key, default)
        return default

    def step(self, task_description: str) -> Dict[str, Any]:
        """
        Execute one operational thought/action cycle.
        Override in specialized agent subclasses.
        """
        self.local_memory.append({"role": "task", "content": task_description})
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "status": "completed",
            "result": f"Agent {self.agent_id} ({self.role.value}) processed: {task_description}",
        }
