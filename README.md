# NanoMind 🧠

A **from-scratch GPT-style transformer language model** (~1–3M parameters) implemented in pure PyTorch. A tiny mind — big ideas. Train it on any text file in minutes.

## Architecture

| Component | Detail |
|---|---|
| Tokenizer | Character-level |
| Embedding | Token + Positional (learned) |
| Normalization | Pre-norm LayerNorm |
| Attention | Multi-head causal self-attention |
| Activation | GELU |
| Regularization | Dropout + weight decay |
| Optimizer | AdamW with cosine LR + warmup |
| Generation | Top-k + temperature sampling |

## Quick Start

### 1. Install Dependencies

```bash
pip install torch
```

### 2. Get Training Data

The classic benchmark is **tiny-shakespeare** (~1MB of Shakespeare plays):

```bash
# The data.txt file is already downloaded if you used the setup script.
# Or download manually:
curl -o data.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### 3. Train

```bash
# Default: 4-layer, 128-dim NanoMind, 5000 iters (~15 min on CPU)
python train.py --data data.txt

# Faster smoke test (100 iters, ~30 seconds on CPU)
python train.py --data data.txt --max_iters 100 --eval_interval 50

# Bigger model (more expressive, needs more training)
python train.py --data data.txt --d_model 256 --n_layers 6 --n_heads 8 --max_iters 10000

# GPU training
python train.py --data data.txt --device cuda --batch_size 64
```

### 4. Generate Text

```bash
# Basic generation
python generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:"

# More tokens, lower temperature (more focused)
python generate.py --checkpoint checkpoints/best.pt \
    --prompt "To be or not to be" \
    --tokens 500 --temperature 0.6 --top_k 20

# Multiple independent samples
python generate.py --checkpoint checkpoints/best.pt \
    --prompt "The king" --num_samples 3
```

## Configuration Reference

### Model (`--` flags in `train.py`)

| Flag | Default | Description |
|---|---|---|
| `--d_model` | 128 | Embedding / hidden dimension |
| `--n_layers` | 4 | Number of transformer blocks |
| `--n_heads` | 4 | Attention heads per block |
| `--block_size` | 128 | Context window (tokens) |
| `--dropout` | 0.1 | Dropout probability |

### Training

| Flag | Default | Description |
|---|---|---|
| `--max_iters` | 5000 | Total gradient steps |
| `--batch_size` | 32 | Sequences per batch |
| `--learning_rate` | 3e-4 | Peak LR |
| `--warmup_iters` | 100 | LR warmup steps |
| `--grad_clip` | 1.0 | Gradient clip norm |
| `--eval_interval` | 200 | Val eval every N steps |
| `--device` | auto | `auto` \| `cpu` \| `cuda` \| `mps` |

### Generation

| Flag | Default | Description |
|---|---|---|
| `--tokens` | 300 | New tokens to generate |
| `--temperature` | 0.8 | > 1 = more random, < 1 = focused |
| `--top_k` | 40 | Restrict sampling to top-K logits |
| `--num_samples` | 1 | Number of independent samples |

## File Structure

```
minigpt/
├── config.py      # ModelConfig + TrainConfig dataclasses
├── tokenizer.py   # CharTokenizer (encode/decode, vocab save/load)
├── model.py       # MiniGPT transformer (attention, FFN, blocks)
├── data.py        # TextDataset + DataLoader helpers
├── train.py       # Training loop CLI
├── generate.py    # Text generation CLI
├── data.txt       # Training corpus (you provide this)
└── checkpoints/
    ├── best.pt    # Best checkpoint by val loss
    ├── latest.pt  # Most recent checkpoint
    └── vocab.json # Saved tokenizer vocabulary
```

## Expected Training Curves — NanoMind on tiny-shakespeare (default config)

| Step | Train Loss | Notes |
|---|---|---|
| 100 | ~2.5 | Model starts learning |
| 500 | ~1.8 | Structure emerging |
| 2000 | ~1.5 | Character patterns learned |
| 5000 | ~1.3 | Coherent Shakespeare-like text |

Cross-entropy of ~1.3 corresponds to the model correctly predicting the next character ~27% of the time — impressive for a ~1M parameter model!

## Example Output — NanoMind after 5000 steps

Trained on tiny-shakespeare:

```
ROMEO:
And so, the night, the beauty of the world,
That I have done, and I will not speak thee;
I will not speak thee in the world of love,
And I will not speak thee in the world of love.
```

## Resuming Training

```bash
python train.py --data data.txt --resume checkpoints/latest.pt --max_iters 10000
```

## License

MIT
