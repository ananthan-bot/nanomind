"""
examples/beam_search_demo.py — Beam search vs. greedy decoding demo.

Compares greedy, standard beam search, and diverse beam search outputs
from the same prompt and model.

Usage:
    python examples/beam_search_demo.py
"""

import torch
from nanomind import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.generate.beam import BeamConfig, beam_search, diverse_beam_search
from nanomind.generate import generate_text

CORPUS    = "the quick brown fox jumps over the lazy dog. " * 40
tokenizer = CharTokenizer().build(CORPUS)
VOCAB     = tokenizer.vocab_size
BLOCK     = 48
device    = torch.device("cpu")

model = NanoMind(ModelConfig(
    vocab_size=VOCAB, block_size=BLOCK,
    d_model=64, n_layers=2, n_heads=4, dropout=0.0,
)).to(device)

PROMPT   = "the quick"
ids      = tokenizer.encode(PROMPT)
idx      = torch.tensor([ids], dtype=torch.long, device=device)
n_prompt = len(ids)

print("=" * 60)
print("Greedy decoding:")
greedy_text = generate_text(model, tokenizer, PROMPT, max_new_tokens=30, strategy="greedy")
print(f"  {greedy_text!r}")

print("\nStandard Beam Search (B=4, length_penalty=1.2):")
cfg  = BeamConfig(num_beams=4, max_new_tokens=30, length_penalty=1.2, return_n_best=4)
hyps = beam_search(model, idx, cfg)
for i, h in enumerate(hyps):
    text = tokenizer.decode(h.tokens[n_prompt:])
    print(f"  Beam {i+1} (score={h.score(1.2):.3f}): {text!r}")

print("\nDiverse Beam Search (B=4, G=2, div_penalty=0.5):")
dcfg = BeamConfig(num_beams=4, max_new_tokens=30, num_beam_groups=2,
                  diversity_penalty=0.5, return_n_best=4)
dhyps = diverse_beam_search(model, idx, dcfg)
for i, h in enumerate(dhyps):
    text = tokenizer.decode(h.tokens[n_prompt:])
    print(f"  Beam {i+1} (score={h.score():.3f}): {text!r}")
print("=" * 60)
