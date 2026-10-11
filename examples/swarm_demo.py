"""
examples/swarm_demo.py — End-to-end Demonstration of Autonomous Multi-Agent Swarms & Compound AI.
Demonstrates:
  1. MessageBroker Pub-Sub Communication
  2. Shared Blackboard Read/Write and Event Watchers
  3. SwarmOrchestrator Multi-Agent Execution (Architect -> Coder -> Reviewer)
  4. Consensus Voting (Majority & Borda Count)
  5. Multi-Agent Adversarial Debate Loop
"""
from nanomind.swarm import (
    SwarmConfig,
    SwarmOrchestrator,
    MessageBroker,
    MessageEnvelope,
    Blackboard,
    ConsensusEngine,
    MultiAgentDebate,
)


def run_demo():
    print("=" * 70)
    print("  NanoMind Day 61: Autonomous Multi-Agent Swarms & Compound AI")
    print("=" * 70)

    # 1. Pub-Sub Message Broker
    print("\n[1] Pub-Sub Message Broker:")
    broker = MessageBroker()
    broker.register_agent("agent_1")
    broker.register_agent("agent_2")

    msg = MessageEnvelope(sender_id="agent_1", recipient_id="agent_2", topic="task", payload="Compute prime numbers")
    broker.publish(msg)
    inbox2 = broker.pop_inbox("agent_2")
    print(f"  Sent message from agent_1 to agent_2 on topic '{msg.topic}'")
    print(f"  agent_2 received: '{inbox2[0].payload}'")

    # 2. Shared Blackboard Working Memory
    print("\n[2] Shared Blackboard Memory & Watchers:")
    blackboard = Blackboard()
    blackboard.watch("status", lambda k, old, new: print(f"  [Blackboard Watcher] '{k}' updated: {old} -> {new}"))
    blackboard.write("status", "in_progress", author_agent_id="agent_1")
    blackboard.write("status", "completed", author_agent_id="agent_2")

    # 3. Swarm Orchestrator Multi-Agent Pipeline
    print("\n[3] Swarm Orchestrator Compound Pipeline:")
    orchestrator = SwarmOrchestrator()
    task = "Build a robust token authentication module"
    res = orchestrator.execute_compound_pipeline(task)
    print(f"  User Task         : {res['user_prompt']}")
    print(f"  Architect Plan    : {res['plan']['components']}")
    print(f"  Coder Code        : {res['artifact']['code']}")
    print(f"  Reviewer Verdict  : {res['review']['feedback']}")
    print(f"  Consensus Result  : {res['consensus_verdict']} (agreement: {res['agreement_rate']*100:.0f}%)")
    print(f"  Total Handoffs    : {res['handoffs_count']}")

    # 4. Consensus Engine Voting
    print("\n[4] Consensus Engine (Majority & Borda Count):")
    votes = ["Option_A", "Option_A", "Option_B", "Option_A"]
    winner, agreement = ConsensusEngine.majority_vote(votes)
    print(f"  Majority Winner   : {winner} ({agreement*100:.1f}% agreement)")

    rankings = [
        ["Model_X", "Model_Y", "Model_Z"],
        ["Model_X", "Model_Z", "Model_Y"],
        ["Model_Y", "Model_X", "Model_Z"],
    ]
    borda_results = ConsensusEngine.borda_count(rankings)
    print(f"  Borda Count Order : {borda_results}")

    # 5. Multi-Agent Debate
    print("\n[5] Multi-Agent Adversarial Debate:")
    debate = MultiAgentDebate(debater_ids=["agent_math", "agent_logic"], max_rounds=2)

    def debater_fn(agent, topic, transcript):
        if agent == "agent_math":
            return "According to arithmetic axioms, statement is true."
        return "Checking edge cases: under zero-division it holds conditionally."

    def arbiter_fn(topic, transcript):
        return "Consensus reached: Statement is valid with boundary conditions acknowledged."

    debate_res = debate.run_debate("Is mathematical theorem XYZ valid?", debater_fn, arbiter_fn)
    print(f"  Debate Rounds     : {debate_res['rounds_completed']}")
    print(f"  Arbiter Verdict   : {debate_res['final_verdict']}")

    print("\n[OK] NanoMind Multi-Agent Swarm demo completed successfully!")


if __name__ == "__main__":
    run_demo()
