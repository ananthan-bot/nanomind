"""
examples/lora_finetune.py — LoRA fine-tuning demo for NanoMind.

Loads a pre-trained model (or trains a tiny one from scratch),
then fine-tunes it on a new text corpus with LoRA.

Usage:
    python examples/lora_finetune.py
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.lora import LoRAConfig, LoRAModel, print_lora_summary

# ── 1. Base model (pre-trained or randomly initialised) ───────────────────────
VOCAB_TEXT = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?" * 10
tokenizer  = CharTokenizer().build(VOCAB_TEXT)

model_cfg = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=32, d_model=64, n_layers=2, n_heads=4, dropout=0.0
)
base_model = NanoMind(model_cfg)
print(f"Base model: {base_model.num_parameters():,} params")

# ── 2. Fine-tuning corpus ─────────────────────────────────────────────────────
FINETUNE_TEXT = "the quick brown fox jumps over the lazy dog " * 30
ids     = tokenizer.encode(FINETUNE_TEXT)
tokens  = torch.tensor(ids)
BLOCK   = model_cfg.block_size
xs = torch.stack([tokens[i:i+BLOCK]     for i in range(len(ids) - BLOCK - 1)])
ys = torch.stack([tokens[i+1:i+BLOCK+1] for i in range(len(ids) - BLOCK - 1)])
loader = DataLoader(TensorDataset(xs, ys), batch_size=8, shuffle=True, drop_last=True)

# ── 3. Wrap with LoRA ─────────────────────────────────────────────────────────
lora_cfg = LoRAConfig(
    r=4,
    alpha=8.0,
    dropout=0.0,
    target_modules=["q_proj", "v_proj"],   # only adapt Q and V projections
)
lora_model = LoRAModel(base_model, lora_cfg)
lora_model.summary()

# ── 4. Fine-tune with LoRA only ───────────────────────────────────────────────
optimizer = torch.optim.AdamW(lora_model.lora_parameters(), lr=3e-3)
for step in range(200):
    x, y = next(iter(loader))
    optimizer.zero_grad()
    _, loss = lora_model(x, y)
    loss.backward()
    optimizer.step()
    if (step + 1) % 50 == 0:
        print(f"step {step+1}/200  loss={loss.item():.4f}")

# ── 5. Save tiny LoRA checkpoint ──────────────────────────────────────────────
lora_model.save("checkpoints/lora_finetune.pt")
print("\nLoRA weights saved (only A/B matrices — very small!)")

# ── 6. Merge for inference ────────────────────────────────────────────────────
lora_model.merge_for_inference()

from nanomind.generate import Generator, GenerationConfig
from nanomind.tokenizer.char import CharTokenizer
gen = Generator(lora_model, tokenizer)
out = gen.generate("the ", GenerationConfig(max_new_tokens=40, strategy="top_k", top_k=10))
print(f"\nGenerated: the {out}")
