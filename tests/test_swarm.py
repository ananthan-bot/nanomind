"""
tests/test_swarm.py — Unit tests for Autonomous Multi-Agent Swarms & Compound AI.
"""
from nanomind.swarm import (
    MessageEnvelope,
    MessageBroker,
    Blackboard,
    HandoffProtocol,
    ConsensusEngine,
    MultiAgentDebate,
    SwarmOrchestrator,
    compute_consensus_agreement_rate,
)


class TestMessageBroker:
    def test_direct_routing(self):
        broker = MessageBroker()
        broker.register_agent("a1")
        broker.register_agent("a2")
        msg = MessageEnvelope(sender_id="a1", recipient_id="a2", topic="test", payload="hello")
        broker.publish(msg)

        inbox_a2 = broker.pop_inbox("a2")
        assert len(inbox_a2) == 1
        assert inbox_a2[0].payload == "hello"

        # Ensure inbox cleared after pop
        assert len(broker.pop_inbox("a2")) == 0

    def test_broadcast(self):
        broker = MessageBroker()
        broker.register_agent("a1")
        broker.register_agent("a2")
        broker.register_agent("a3")
        msg = MessageEnvelope(sender_id="a1", recipient_id="broadcast", topic="alert", payload="all")
        broker.publish(msg)

        assert len(broker.pop_inbox("a2")) == 1
        assert len(broker.pop_inbox("a3")) == 1
        assert len(broker.pop_inbox("a1")) == 0
