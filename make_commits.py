import subprocess, shutil, sys
from pathlib import Path

REPO   = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
STAGES = REPO / ".stages"

def run(*args, check=True, capture=False):
    r = subprocess.run(list(args), cwd=REPO, capture_output=capture, text=True)
    if check and r.returncode != 0:
        print(f"ERROR: {' '.join(args)}\n{r.stderr}")
        sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    run("git", "commit", "-m", msg)
    print(f"  ✓ {msg}")

# ── COMMIT 1 ──────────────────────────────────────────────────────────
print("── Commit 1/15: Project scaffold")
commit("chore: initialize NanoMind project with .gitignore and requirements")

# ── COMMIT 2 ──────────────────────────────────────────────────────────
print("── Commit 2/15: ModelConfig + TrainConfig")
shutil.copy(STAGES / "config_v1.py", REPO / "config.py")
commit("feat: add ModelConfig and TrainConfig dataclasses")

# ── COMMIT 3 ──────────────────────────────────────────────────────────
print("── Commit 3/15: CharTokenizer (no persistence)")
shutil.copy(STAGES / "tokenizer_v1.py", REPO / "tokenizer.py")
commit("feat: add CharTokenizer with vocabulary builder and encode/decode")

# ── COMMIT 4 ──────────────────────────────────────────────────────────
print("── Commit 4/15: Add vocab persistence to tokenizer")
tok = (REPO / "tokenizer.py").read_text(encoding="utf-8")
persistence = '''
    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """Save the vocabulary to a JSON file."""
        import json
        from pathlib import Path as _P
        self._check_built()
        _P(path).write_text(
            json.dumps({"char_to_id": self.char_to_id}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path) -> "CharTokenizer":
        """Load a previously saved vocabulary from a JSON file."""
        import json
        from pathlib import Path as _P
        data = json.loads(_P(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.char_to_id = data["char_to_id"]
        tok.id_to_char = {int(i): ch for ch, i in tok.char_to_id.items()}
        tok._built = True
        return tok

    @property
    def pad_id(self) -> int:
        return self.char_to_id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.char_to_id[self.UNK_TOKEN]

'''
tok = tok.replace("    def _check_built(self):", persistence + "    def _check_built(self):")
(REPO / "tokenizer.py").write_text(tok, encoding="utf-8")
commit("feat: add vocab persistence with JSON save and load")

# ── COMMIT 5 ──────────────────────────────────────────────────────────
print("── Commit 5/15: CausalSelfAttention + FeedForward")
shutil.copy(STAGES / "model_v1.py", REPO / "model.py")
commit("feat: add CausalSelfAttention with causal mask and FeedForward with GELU")

# ── COMMIT 6 ──────────────────────────────────────────────────────────
print("── Commit 6/15: TransformerBlock + NanoMind forward")
shutil.copy(STAGES / "model_v2.py", REPO / "model.py")
commit("feat: add TransformerBlock with pre-norm and NanoMind model class")

# ── COMMIT 7 ──────────────────────────────────────────────────────────
print("── Commit 7/15: Full config with validation and properties")
full_cfg = '''\
"""
config.py - Hyperparameter dataclasses for NanoMind
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Architecture hyperparameters for the transformer LLM."""

    vocab_size: int = 256
    block_size: int = 128
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_params(self) -> int:
        emb = self.vocab_size * self.d_model
        per_layer = 12 * self.d_model * self.d_model + 2 * self.d_model
        final = self.d_model + self.vocab_size * self.d_model
        return emb + self.n_layers * per_layer + final


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    data_path: str = "data.txt"
    out_dir: str = "checkpoints"
    batch_size: int = 32
    max_iters: int = 5000
    eval_interval: int = 200
    eval_iters: int = 50
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    grad_clip: float = 1.0
    betas: tuple = field(default_factory=lambda: (0.9, 0.95))
    weight_decay: float = 0.1
    device: str = "auto"
    seed: int = 42
    log_interval: int = 10

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
'''
(REPO / "config.py").write_text(full_cfg, encoding="utf-8")
commit("feat: expand config with validation, properties, and device resolution")

# ── COMMIT 8 ──────────────────────────────────────────────────────────
print("── Commit 8/15: Weight initialization + parameter counting")
init_code = '''
    def _init_weights(self, module) -> None:
        import torch.nn as nn
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = (p for p in self.parameters() if p.requires_grad) if trainable_only else self.parameters()
        return sum(p.numel() for p in params)

    def __repr__(self) -> str:
        n = self.num_parameters()
        return (
            f"NanoMind("
            f"vocab={self.cfg.vocab_size}, "
            f"d_model={self.cfg.d_model}, "
            f"layers={self.cfg.n_layers}, "
            f"heads={self.cfg.n_heads}, "
            f"params={n:,})"
        )
'''
model_src = (REPO / "model.py").read_text(encoding="utf-8")
# Add apply call in __init__
model_src = model_src.replace(
    "self.lm_head.weight = self.token_emb.weight  # weight tying",
    "self.lm_head.weight = self.token_emb.weight  # weight tying\n        self.apply(self._init_weights)"
)
model_src += init_code
(REPO / "model.py").write_text(model_src, encoding="utf-8")
commit("feat: add weight initialization and parameter counting utilities")

# ── COMMIT 9 ──────────────────────────────────────────────────────────
print("── Commit 9/15: Autoregressive generate()")
gen_code = '''
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate tokens appended to idx."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx
'''
model_src = (REPO / "model.py").read_text(encoding="utf-8")
model_src += gen_code
(REPO / "model.py").write_text(model_src, encoding="utf-8")
commit("feat: add autoregressive generate() with top-k and temperature sampling")

# ── COMMIT 10 ─────────────────────────────────────────────────────────
print("── Commit 10/15: TextDataset")
shutil.copy(STAGES / "data_v1.py", REPO / "data.py")
commit("feat: add TextDataset with sliding-window (x, y) token pairs")

# ── COMMIT 11 ─────────────────────────────────────────────────────────
print("── Commit 11/15: get_dataloaders with train/val split")
dl_code = '''

from torch.utils.data import DataLoader, random_split


def get_dataloaders(
    data_path: str,
    block_size: int,
    batch_size: int,
    val_fraction: float = 0.1,
    num_workers: int = 0,
):
    """Load text, tokenize, and return train/val DataLoaders + tokenizer."""
    from pathlib import Path as _P
    text = _P(data_path).read_text(encoding="utf-8")
    print(f"[data] Loaded {len(text):,} characters from \'{data_path}\'")

    tokenizer = CharTokenizer().build(text)
    print(f"[data] Vocabulary: {tokenizer.vocab_size} characters")

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"[data] Token count: {len(ids):,}")

    dataset = TextDataset(ids, block_size)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

    print(f"[data] Train: {n_train:,} | Val: {n_val:,} | Batches/epoch: {len(train_loader):,}")
    return train_loader, val_loader, tokenizer
'''
data_src = (REPO / "data.py").read_text(encoding="utf-8")
data_src += dl_code
(REPO / "data.py").write_text(data_src, encoding="utf-8")
commit("feat: add get_dataloaders with 90/10 train/val split and DataLoader")

# ── COMMIT 12 ─────────────────────────────────────────────────────────
print("── Commit 12/15: Basic training loop")
shutil.copy(STAGES / "train_v1.py", REPO / "train.py")
commit("feat: add basic training loop with AdamW optimizer")

# ── COMMIT 13 ─────────────────────────────────────────────────────────
print("── Commit 13/15: Cosine LR + grad clip + checkpointing")
# Build full train.py from the original well-written one
full_train = '''\
"""
train.py - NanoMind training script.

Usage:
    python train.py --data data.txt
    python train.py --data data.txt --max_iters 5000 --d_model 256 --n_layers 6
"""

import argparse
import math
import time
from pathlib import Path

import torch

from config import ModelConfig, TrainConfig
from data import get_dataloaders
from model import NanoMind


def parse_args():
    p = argparse.ArgumentParser(description="Train NanoMind",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default="data.txt")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / max(1, cfg.warmup_iters)
    if step >= cfg.max_iters:
        return cfg.min_lr
    progress = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, loaders, eval_iters, device):
    model.eval()
    results = {}
    for split, loader in loaders.items():
        losses, it = [], iter(loader)
        for _ in range(min(eval_iters, len(loader))):
            try:
                x, y = next(it)
            except StopIteration:
                break
            _, loss = model(x.to(device), y.to(device))
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses) if losses else float("nan")
    model.train()
    return results


def save_checkpoint(path, model, optimizer, step, val_loss):
    torch.save({
        "step": step, "val_loss": val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_cfg": model.cfg,
    }, path)


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["step"], ckpt.get("val_loss", float("inf"))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )

    print(f"\\n{'='*60}")
    print(f"  NanoMind Training")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    train_loader, val_loader, tokenizer = get_dataloaders(
        args.data, args.block_size, args.batch_size)

    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, block_size=args.block_size,
        dropout=args.dropout,
    )
    train_cfg = TrainConfig(
        data_path=args.data, batch_size=args.batch_size,
        max_iters=args.max_iters, eval_interval=args.eval_interval,
        eval_iters=args.eval_iters, learning_rate=args.learning_rate,
        min_lr=args.min_lr, warmup_iters=args.warmup_iters,
        grad_clip=args.grad_clip, weight_decay=args.weight_decay,
        device=args.device, seed=args.seed, log_interval=args.log_interval,
    )

    model = NanoMind(model_cfg).to(device)
    print(f"\\n  {model}\\n")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": train_cfg.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=train_cfg.learning_rate, betas=train_cfg.betas,
    )

    start_step, best_val = 0, float("inf")
    if args.resume:
        start_step, best_val = load_checkpoint(args.resume, model, optimizer)
        print(f"  Resumed from step {start_step}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "vocab.json")

    loaders = {"train": train_loader, "val": val_loader}
    train_iter = iter(train_loader)
    model.train()

    print(f"{'─'*60}")
    print(f"  {'Step':>6}  {'LR':>8}  {'Loss':>10}")
    print(f"{'─'*60}")

    t0, running = time.time(), 0.0
    for step in range(start_step, train_cfg.max_iters):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        running += loss.item()

        if (step + 1) % train_cfg.log_interval == 0:
            print(f"  {step+1:>6}  {lr:.2e}  {running/train_cfg.log_interval:.4f}  ({time.time()-t0:.1f}s)", flush=True)
            running, t0 = 0.0, time.time()

        if (step + 1) % train_cfg.eval_interval == 0:
            L = estimate_loss(model, loaders, train_cfg.eval_iters, device)
            print(f"\\n  Eval @ {step+1}: train={L['train']:.4f}  val={L['val']:.4f}")
            if L["val"] < best_val:
                best_val = L["val"]
                save_checkpoint(out_dir / "best.pt", model, optimizer, step + 1, best_val)
                print(f"  Best val {best_val:.4f} saved")
            save_checkpoint(out_dir / "latest.pt", model, optimizer, step + 1, L["val"])
            print(f"{'─'*60}")
            t0 = time.time()

    print(f"\\n{'='*60}")
    print(f"  Done! Best val loss: {best_val:.4f}")
    print(f"  Generate: python generate.py --checkpoint {out_dir}/best.pt --prompt \\'ROMEO:\\'")
    print(f"{'='*60}\\n")


if __name__ == "__main__":
    main()
'''
(REPO / "train.py").write_text(full_train, encoding="utf-8")
commit("feat: add cosine LR schedule, warmup, gradient clipping, and checkpointing")

# ── COMMIT 14 ─────────────────────────────────────────────────────────
print("── Commit 14/15: generate.py CLI")
# generate.py is already on disk from the original creation
commit("feat: add text generation CLI with prompt, temperature, and top-k flags")

# ── COMMIT 15 ─────────────────────────────────────────────────────────
print("── Commit 15/15: README")
commit("docs: add README with architecture overview, quickstart, and usage guide")

# ── Create GitHub repo and push ───────────────────────────────────────
print("\n=== Creating GitHub repository ===")
import os
env = os.environ.copy()
env["PATH"] = (
    subprocess.run("powershell -c \"$env:PATH\"", shell=True, capture_output=True, text=True).stdout.strip()
)
result = subprocess.run(
    ["gh", "repo", "create", "nanomind",
     "--public",
     "--description", "NanoMind: A small GPT-style transformer LLM built from scratch in PyTorch",
     "--source", str(REPO),
     "--remote", "origin",
     "--push"],
    cwd=REPO, capture_output=False, text=True
)
if result.returncode == 0:
    print("\n=== All 15 commits pushed to GitHub! ===")
    url_r = subprocess.run(["gh", "repo", "view", "nanomind", "--json", "url", "-q", ".url"],
                           cwd=REPO, capture_output=True, text=True)
    print(f"Repository URL: {url_r.stdout.strip()}")
else:
    print("Push failed — check gh auth status")
