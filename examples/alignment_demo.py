"""
examples/alignment_demo.py — NanoMind Constitutional AI, DPO & Alignment demo.

Demonstrates:
  1. Preference dataset: pairs, filtering, stats
  2. DPO loss: DPO / IPO / KTO-pair variants
  3. DPO trainer: policy + frozen reference
  4. Constitutional AI: critique-revision, RLAIF
  5. KTO: binary feedback training
  6. Reward model: Bradley-Terry preference loss
  7. Alignment evaluation: win rate, KL divergence

Usage:
    python examples/alignment_demo.py
"""
import torch
import torch.nn as nn
from nanomind.alignment import (
    PreferencePair, BinaryFeedback, PreferenceDataset,
    DPOConfig, dpo_loss, compute_log_probs, DPOTrainer,
    ConstitutionalAI, ConstitutionalConfig, CritiqueRevision,
    KTOConfig, kto_loss, KTOTrainer,
    RewardModel, RMTrainingConfig,
    AlignmentEvaluator, AlignmentMetrics,
)

V = 32   # small vocab for demo

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 16)
        self.rnn  = nn.GRU(16, 32, batch_first=True)
        self.head = nn.Linear(32, V)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h), None

print("=" * 60)
print("NanoMind Constitutional AI, DPO & Alignment Demo")
print("=" * 60)

# ── Preference Dataset ────────────────────────────────────────────────────────
print("
── Preference Dataset ──")
dataset = PreferenceDataset([
    PreferencePair("What is 2+2?", "4", "fish",
                    score_chosen=0.9, score_rejected=0.1, source="human"),
    PreferencePair("Tell me a joke", "Why did the chicken cross the road?",
                    "I refuse to answer", score_chosen=0.8, score_rejected=0.2,
                    source="human"),
    PreferencePair("What is AI?", "Artificial Intelligence is...", "IDK",
                    score_chosen=0.7, score_rejected=0.3, source="ai"),
])
print(f"  Dataset size: {len(dataset)}")
print(f"  Stats: {dataset.stats()}")
filtered = dataset.filter_by_margin(min_margin=0.5)
print(f"  After filter(margin≥0.5): {len(filtered)} pairs")
binary = dataset.to_binary()
print(f"  Binary feedback samples: {len(binary)}")

# ── DPO Loss ──────────────────────────────────────────────────────────────────
print("
── DPO Loss (Direct Preference Optimization) ──")
B = 4
policy_logp_w = torch.tensor([-2.1, -1.8, -2.5, -1.9])
policy_logp_l = torch.tensor([-3.2, -3.5, -3.8, -2.8])
ref_logp_w    = torch.tensor([-2.0, -2.0, -2.4, -2.0])
ref_logp_l    = torch.tensor([-3.0, -3.2, -3.5, -2.7])

for loss_type in ["dpo", "ipo", "kto_pair"]:
    cfg  = DPOConfig(beta=0.1, loss_type=loss_type)
    loss, m = dpo_loss(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, cfg)
    print(f"  [{loss_type:8}] loss={m['loss']:.4f} acc={m['accuracy']:.2%} "
          f"margin={m['reward_margin']:.4f}")

# ── DPO Trainer ───────────────────────────────────────────────────────────────
print("
── DPO Trainer ──")
policy = TinyLM(); ref = TinyLM()
opt    = torch.optim.AdamW(policy.parameters(), lr=1e-4)
trainer = DPOTrainer(policy, ref, opt, DPOConfig(beta=0.1))
chosen_ids   = torch.randint(0, V, (2, 8))
rejected_ids = torch.randint(0, V, (2, 8))
metrics = trainer.step(chosen_ids, rejected_ids)
print(f"  DPO step: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.2%}")
print(f"  Chosen reward: {metrics['reward_chosen']:.4f}")
print(f"  Rejected reward: {metrics['reward_rejected']:.4f}")

# ── Constitutional AI ──────────────────────────────────────────────────────────
print("
── Constitutional AI ──")
cai = ConstitutionalAI()
cr  = cai.critique_and_revise(
    response  = "This question is stupid and I won't answer it.",
    principle = "Choose the response that is most respectful of human dignity.",
)
print(f"  Original: {cr.original_response[:50]}...")
print(f"  Critique: {cr.critique[:50]}...")
print(f"  Revised:  {cr.revised_response[:50]}...")

# RLAIF: AI feedback on response pairs
print("
  RLAIF — AI Feedback:")
preferred = cai.ai_feedback(
    "What is the capital of France?",
    "Paris is the capital of France.",
    "I don't know.",
)
print(f"  Preferred response: {preferred}")

# Build preference dataset via RLAIF
rlaif_dataset = cai.build_preference_dataset(
    prompts     = ["What is 2+2?", "Explain gravity"],
    responses_a = ["The answer is 4.", "Gravity is a force..."],
    responses_b = ["I don't know", "It's complicated."],
)
print(f"  RLAIF dataset size: {len(rlaif_dataset)}")
print(f"  RLAIF stats: {rlaif_dataset.stats()}")

# SFT data generation
sft_data = cai.generate_sft_data(
    ["How are you?"], ["I am functioning."]
)
print(f"  SFT data generated: {sft_data[0].keys()}")

# ── KTO ───────────────────────────────────────────────────────────────────────
print("
── KTO (Kahneman-Tversky Optimization) ──")
policy2 = TinyLM(); ref2 = TinyLM()
opt2    = torch.optim.AdamW(policy2.parameters(), lr=1e-4)
kto_trainer = KTOTrainer(policy2, ref2, opt2, KTOConfig(beta=0.1))
ids        = torch.randint(0, V, (4, 8))
is_good    = torch.tensor([True, False, True, False])
kto_m      = kto_trainer.step(ids, is_good)
print(f"  KTO step: loss={kto_m['loss']:.4f}")
print(f"  Reward desirable: {kto_m['reward_desirable']:.4f}")
print(f"  Reward undesirable: {kto_m['reward_undesirable']:.4f}")

# Direct kto_loss
policy_lp = torch.tensor([-2.0, -3.0, -2.5, -3.5])
ref_lp    = torch.tensor([-2.1, -2.9, -2.4, -3.3])
kl_est    = torch.tensor(0.1)
cfg_kto   = KTOConfig(beta=0.1)
l, m      = kto_loss(policy_lp, ref_lp, is_good, kl_est, cfg_kto)
print(f"  KTO loss: {l.item():.4f}")

# ── Reward Model ──────────────────────────────────────────────────────────────
print("
── Reward Model (Bradley-Terry) ──")
rm     = RewardModel(TinyLM(), d_model=V)
chosen = torch.randint(0, V, (4, 8))
reject = torch.randint(0, V, (4, 8))
rewards = rm(chosen)
print(f"  Reward scores shape: {tuple(rewards.shape)}")
print(f"  Mean reward: {rewards.mean().item():.4f}")
rm_loss, rm_m = rm.preference_loss(chosen, reject)
print(f"  RM loss: {rm_m['loss']:.4f}, acc: {rm_m['accuracy']:.2%}")
print(f"  Reward margin: {rm_m['reward_margin']:.4f}")

# ── Alignment Evaluator ────────────────────────────────────────────────────────
print("
── Alignment Evaluation ──")
evaluator = AlignmentEvaluator(rm, TinyLM())
test_ids  = torch.randint(0, V, (8, 10))
metrics   = evaluator.evaluate(policy, ref, test_ids)
print(f"  Win rate:     {metrics.win_rate:.2%}")
print(f"  Mean reward:  {metrics.mean_reward:.4f}")
print(f"  KL divergence:{metrics.kl_div:.4f}")
print(f"  Reward hacking: {metrics.is_reward_hacking()}")
print(f"  Metrics: {metrics.to_dict()}")

print("
Alignment demo complete!")
