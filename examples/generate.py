"""
examples/generate.py — Text generation demo from a NanoMind checkpoint.

Usage:
    python examples/generate.py --checkpoint checkpoints/tiny/best.pt \
                                 --prompt "The quick"
"""

import argparse, torch
from nanomind.checkpoint import load_for_inference, checkpoint_info
from nanomind.model import NanoMind, ModelConfig
from nanomind.tokenizer.char import CharTokenizer
from nanomind.generate import Generator, GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--prompt", default="The ")
parser.add_argument("--max-new-tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=40)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

meta       = checkpoint_info(args.checkpoint)
model_cfg  = ModelConfig(**meta.get("model_config", {}))
model      = NanoMind(model_cfg).to(device)
load_for_inference(args.checkpoint, model, device=device)

tokenizer  = CharTokenizer()
generator  = Generator(model, tokenizer, device=device)
gen_cfg    = GenerationConfig(
    max_new_tokens=args.max_new_tokens,
    strategy="top_k",
    top_k=args.top_k,
    temperature=args.temperature,
)
print(args.prompt + generator.generate(args.prompt, gen_cfg))
