"""
nanomind/swarm/message.py — Typed message envelopes and publish-subscribe message broker.
"""
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import time
import uuid


@dataclass
class MessageEnvelope:
    """Standardized inter-agent communication packet."""
    sender_id: str
    recipient_id: str  # Specific agent ID or "broadcast"
    topic: str
    payload: Any
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None


class MessageBroker:
    """
    Central in-memory publish-subscribe message router.
    Routes messages to target agent inboxes and dispatches broadcast events.
    """

    def __init__(self):
        # agent_id -> list of received messages
        self.inboxes: Dict[str, List[MessageEnvelope]] = {}
        # topic -> list of subscriber callback handlers
        self.topic_subscribers: Dict[str, List[Callable[[MessageEnvelope], None]]] = {}
        self.message_history: List[MessageEnvelope] = []

    def register_agent(self, agent_id: str):
        if agent_id not in self.inboxes:
            self.inboxes[agent_id] = []

    def subscribe(self, topic: str, handler: Callable[[MessageEnvelope], None]):
        if topic not in self.topic_subscribers:
            self.topic_subscribers[topic] = []
        self.topic_subscribers[topic].append(handler)

    def publish(self, message: MessageEnvelope):
        """Route message to recipient inbox or broadcast across swarm."""
        self.message_history.append(message)

        # 1. Topic subscriptions
        if message.topic in self.topic_subscribers:
            for handler in self.topic_subscribers[message.topic]:
                handler(message)

        # 2. Direct routing or broadcast
        if message.recipient_id == "broadcast":
            for agent_id, box in self.inboxes.items():
                if agent_id != message.sender_id:
                    box.append(message)
        elif message.recipient_id in self.inboxes:
            self.inboxes[message.recipient_id].append(message)

    def pop_inbox(self, agent_id: str) -> List[MessageEnvelope]:
        """Fetch and clear all pending messages for an agent."""
        if agent_id not in self.inboxes:
            return []
        msgs = list(self.inboxes[agent_id])
        self.inboxes[agent_id].clear()
        return msgs
