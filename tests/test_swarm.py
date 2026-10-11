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


class TestBlackboard:
    def test_read_write(self):
        bb = Blackboard()
        bb.write("key1", "val1", author_agent_id="a1")
        assert bb.read("key1") == "val1"
        assert bb.read("nonexistent", default="def") == "def"

    def test_watchers(self):
        bb = Blackboard()
        captured = []
        bb.watch("k", lambda k, old, new: captured.append((old, new)))
        bb.write("k", "v1", author_agent_id="a1")
        bb.write("k", "v2", author_agent_id="a2")
        assert len(captured) == 2
        assert captured[0] == (None, "v1")
        assert captured[1] == ("v1", "v2")


class TestHandoffProtocol:
    def test_handoff_execution(self):
        protocol = HandoffProtocol()
        ctx = protocol.execute_handoff(
            source_agent_id="planner",
            target_agent_id="coder",
            reason="Delegate coding",
            task_state={"step": 1},
        )
        assert ctx.source_agent_id == "planner"
        assert ctx.target_agent_id == "coder"
        assert len(protocol.get_history()) == 1


class TestConsensusEngine:
    def test_majority_voting(self):
        votes = ["A", "B", "A", "A"]
        winner, frac = ConsensusEngine.majority_vote(votes)
        assert winner == "A"
        assert frac == 0.75

    def test_weighted_consensus(self):
        votes = [("A", 1.0), ("B", 4.0), ("A", 1.0)]
        winner, frac = ConsensusEngine.weighted_consensus(votes)
        assert winner == "B"
        assert frac == 4.0 / 6.0

    def test_borda_count(self):
        rankings = [["A", "B", "C"], ["B", "A", "C"]]
        # A: 3 + 2 = 5, B: 2 + 3 = 5, C: 1 + 1 = 2
        res = ConsensusEngine.borda_count(rankings)
        assert res[0][1] == 5
        assert res[-1][0] == "C"
