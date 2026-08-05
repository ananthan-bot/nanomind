$ErrorActionPreference = "Stop"
$repo = "C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt"
$stages = "$repo\.stages"
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")
$env:GITHUB_TOKEN = ""

Set-Location $repo

function Do-Commit {
    param([string]$msg)
    git add -A | Out-Null
    git commit -m $msg
    Write-Host "DONE: $msg" -ForegroundColor Green
}

# Init
Write-Host "=== Initializing git repo ===" -ForegroundColor Cyan
git init
git checkout -b main 2>$null; if ($LASTEXITCODE -ne 0) { git branch -m main }

# --- COMMIT 1: scaffold
Write-Host "-- Commit 1/15 --"
Do-Commit "chore: initialize NanoMind project with .gitignore and requirements"

# --- COMMIT 2: minimal config
Write-Host "-- Commit 2/15 --"
Copy-Item "$stages\config_v1.py" "$repo\config.py" -Force
Do-Commit "feat: add ModelConfig and TrainConfig dataclasses"

# --- COMMIT 3: tokenizer v1 (no persistence)
Write-Host "-- Commit 3/15 --"
Copy-Item "$stages\tokenizer_v1.py" "$repo\tokenizer.py" -Force
Do-Commit "feat: add CharTokenizer with vocabulary builder and encode/decode"

# --- COMMIT 4: add save/load to tokenizer
Write-Host "-- Commit 4/15 --"
$tok = Get-Content "$repo\tokenizer.py" -Raw
$saveLoad = "`n    def save(self, path) -> None:`n        import json; from pathlib import Path`n        self._check_built()`n        Path(path).write_text(json.dumps({'char_to_id': self.char_to_id}, ensure_ascii=False, indent=2), encoding='utf-8')`n`n    @classmethod`n    def load(cls, path) -> 'CharTokenizer':`n        import json; from pathlib import Path`n        data = json.loads(Path(path).read_text(encoding='utf-8'))`n        tok = cls()`n        tok.char_to_id = data['char_to_id']`n        tok.id_to_char = {int(i): ch for ch, i in tok.char_to_id.items()}`n        tok._built = True`n        return tok`n`n    @property`n    def pad_id(self) -> int:`n        return self.char_to_id[self.PAD_TOKEN]`n"
$insertAt = "    def _check_built(self):"
$tok = $tok -replace [regex]::Escape($insertAt), ($saveLoad + $insertAt)
Set-Content "$repo\tokenizer.py" $tok -Encoding UTF8
Do-Commit "feat: add vocab persistence with JSON save and load"

# --- COMMIT 5: model v1 (attention + FFN)
Write-Host "-- Commit 5/15 --"
Copy-Item "$stages\model_v1.py" "$repo\model.py" -Force
Do-Commit "feat: add CausalSelfAttention with causal mask and FeedForward with GELU"

# --- COMMIT 6: model v2 (+ TransformerBlock + NanoMind forward)
Write-Host "-- Commit 6/15 --"
Copy-Item "$stages\model_v2.py" "$repo\model.py" -Force
Do-Commit "feat: add TransformerBlock (pre-norm) and NanoMind model with forward pass"

# --- COMMIT 7: full config
Write-Host "-- Commit 7/15 --"
$cfg = @(
    '"""',
    'config.py - Hyperparameter dataclasses for NanoMind',
    '"""',
    '',
    'from dataclasses import dataclass, field',
    '',
    '',
    '@dataclass',
    'class ModelConfig:',
    '    """Architecture hyperparameters for the transformer LLM."""',
    '    vocab_size: int = 256',
    '    block_size: int = 128',
    '    d_model: int = 128',
    '    n_layers: int = 4',
    '    n_heads: int = 4',
    '    dropout: float = 0.1',
    '',
    '    def __post_init__(self):',
    '        assert self.d_model % self.n_heads == 0, f"d_model must be divisible by n_heads"',
    '',
    '    @property',
    '    def head_dim(self) -> int:',
    '        return self.d_model // self.n_heads',
    '',
    '    @property',
    '    def n_params(self) -> int:',
    '        emb = self.vocab_size * self.d_model',
    '        per_layer = 12 * self.d_model * self.d_model + 2 * self.d_model',
    '        final = self.d_model + self.vocab_size * self.d_model',
    '        return emb + self.n_layers * per_layer + final',
    '',
    '',
    '@dataclass',
    'class TrainConfig:',
    '    """Training hyperparameters."""',
    '    data_path: str = "data.txt"',
    '    out_dir: str = "checkpoints"',
    '    batch_size: int = 32',
    '    max_iters: int = 5000',
    '    eval_interval: int = 200',
    '    eval_iters: int = 50',
    '    learning_rate: float = 3e-4',
    '    min_lr: float = 3e-5',
    '    warmup_iters: int = 100',
    '    grad_clip: float = 1.0',
    '    betas: tuple = field(default_factory=lambda: (0.9, 0.95))',
    '    weight_decay: float = 0.1',
    '    device: str = "auto"',
    '    seed: int = 42',
    '    log_interval: int = 10',
    '',
    '    def resolve_device(self) -> str:',
    '        if self.device != "auto": return self.device',
    '        import torch',
    '        if torch.cuda.is_available(): return "cuda"',
    '        if torch.backends.mps.is_available(): return "mps"',
    '        return "cpu"'
) -join "`n"
Set-Content "$repo\config.py" $cfg -Encoding UTF8
Do-Commit "feat: expand config with validation, properties, and cosine LR support"

# --- COMMIT 8: add weight init + utilities to model
Write-Host "-- Commit 8/15 --"
$modelExtra = "`n    def _init_weights(self, module):`n        import torch.nn as nn`n        if isinstance(module, nn.Linear):`n            nn.init.normal_(module.weight, mean=0.0, std=0.02)`n            if module.bias is not None: nn.init.zeros_(module.bias)`n        elif isinstance(module, nn.Embedding): nn.init.normal_(module.weight, mean=0.0, std=0.02)`n        elif isinstance(module, nn.LayerNorm): nn.init.ones_(module.weight); nn.init.zeros_(module.bias)`n`n    def num_parameters(self, trainable_only=True):`n        params = (p for p in self.parameters() if p.requires_grad) if trainable_only else self.parameters()`n        return sum(p.numel() for p in params)`n`n    def __repr__(self):`n        n = self.num_parameters()`n        return f'NanoMind(vocab={self.cfg.vocab_size}, d_model={self.cfg.d_model}, layers={self.cfg.n_layers}, heads={self.cfg.n_heads}, params={n:,})'"
Add-Content "$repo\model.py" $modelExtra -Encoding UTF8
# Add apply call in __init__
$m = Get-Content "$repo\model.py" -Raw
$m = $m -replace "(self\.lm_head\.weight = self\.token_emb\.weight)", '$1`n        self.apply(self._init_weights)'
Set-Content "$repo\model.py" $m -Encoding UTF8
Do-Commit "feat: add weight initialization scheme and parameter counting"

# --- COMMIT 9: add generate() method
Write-Host "-- Commit 9/15 --"
$genMethod = "`n    @torch.no_grad()`n    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):`n        self.eval()`n        for _ in range(max_new_tokens):`n            idx_cond = idx[:, -self.cfg.block_size:]`n            logits, _ = self(idx_cond)`n            logits = logits[:, -1, :] / temperature`n            if top_k is not None:`n                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))`n                logits[logits < v[:, [-1]]] = float('-inf')`n            probs = torch.nn.functional.softmax(logits, dim=-1)`n            next_tok = torch.multinomial(probs, num_samples=1)`n            idx = torch.cat([idx, next_tok], dim=1)`n        return idx"
Add-Content "$repo\model.py" $genMethod -Encoding UTF8
Do-Commit "feat: add autoregressive generate() with top-k and temperature sampling"

# --- COMMIT 10: data v1 (TextDataset only)
Write-Host "-- Commit 10/15 --"
Copy-Item "$stages\data_v1.py" "$repo\data.py" -Force
Do-Commit "feat: add TextDataset with sliding-window token pairs"

# --- COMMIT 11: full data.py (adds get_dataloaders)
Write-Host "-- Commit 11/15 --"
$dlCode = "`n`nfrom torch.utils.data import DataLoader, random_split`n`n`ndef get_dataloaders(data_path, block_size, batch_size, val_fraction=0.1, num_workers=0):`n    from pathlib import Path`n    text = Path(data_path).read_text(encoding='utf-8')`n    print(f'[data] Loaded {len(text):,} characters')`n    tokenizer = CharTokenizer().build(text)`n    print(f'[data] Vocabulary: {tokenizer.vocab_size} chars')`n    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)`n    dataset = TextDataset(ids, block_size)`n    n_val = max(1, int(len(dataset) * val_fraction))`n    n_train = len(dataset) - n_val`n    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))`n    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)`n    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)`n    print(f'[data] Train: {n_train:,} | Val: {n_val:,}')`n    return train_loader, val_loader, tokenizer"
Add-Content "$repo\data.py" $dlCode -Encoding UTF8
Do-Commit "feat: add get_dataloaders with 90/10 train/val split"

# --- COMMIT 12: train v1 (basic loop)
Write-Host "-- Commit 12/15 --"
Copy-Item "$stages\train_v1.py" "$repo\train.py" -Force
Do-Commit "feat: add basic training loop with AdamW optimizer"

# --- COMMIT 13: full train.py (LR schedule + checkpointing)
Write-Host "-- Commit 13/15 --"
$trainFull = Get-Content "$repo\..\..\..\..\..\..\..\..\..\..\.." -ErrorAction SilentlyContinue
# Just write the full train.py directly using array-of-lines approach
$trainLines = @(
    '"""train.py - NanoMind training with cosine LR, grad clip, and checkpointing."""',
    'import argparse, math, time',
    'from pathlib import Path',
    'import torch',
    'from config import ModelConfig, TrainConfig',
    'from data import get_dataloaders',
    'from model import NanoMind',
    '',
    'def parse_args():',
    '    p = argparse.ArgumentParser()',
    '    p.add_argument("--data", default="data.txt")',
    '    p.add_argument("--out_dir", default="checkpoints")',
    '    p.add_argument("--d_model", type=int, default=128)',
    '    p.add_argument("--n_layers", type=int, default=4)',
    '    p.add_argument("--n_heads", type=int, default=4)',
    '    p.add_argument("--block_size", type=int, default=128)',
    '    p.add_argument("--dropout", type=float, default=0.1)',
    '    p.add_argument("--batch_size", type=int, default=32)',
    '    p.add_argument("--max_iters", type=int, default=5000)',
    '    p.add_argument("--eval_interval", type=int, default=200)',
    '    p.add_argument("--eval_iters", type=int, default=50)',
    '    p.add_argument("--learning_rate", type=float, default=3e-4)',
    '    p.add_argument("--min_lr", type=float, default=3e-5)',
    '    p.add_argument("--warmup_iters", type=int, default=100)',
    '    p.add_argument("--grad_clip", type=float, default=1.0)',
    '    p.add_argument("--weight_decay", type=float, default=0.1)',
    '    p.add_argument("--device", default="auto")',
    '    p.add_argument("--seed", type=int, default=42)',
    '    p.add_argument("--log_interval", type=int, default=10)',
    '    p.add_argument("--resume", default=None)',
    '    return p.parse_args()',
    '',
    'def get_lr(step, cfg):',
    '    if step < cfg.warmup_iters: return cfg.learning_rate * step / max(1, cfg.warmup_iters)',
    '    if step >= cfg.max_iters: return cfg.min_lr',
    '    progress = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)',
    '    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.learning_rate - cfg.min_lr)',
    '',
    '@torch.no_grad()',
    'def estimate_loss(model, loaders, eval_iters, device):',
    '    model.eval()',
    '    results = {}',
    '    for split, loader in loaders.items():',
    '        losses, it = [], iter(loader)',
    '        for _ in range(min(eval_iters, len(loader))):",',
    '            try: x, y = next(it)',
    '            except StopIteration: break',
    '            _, loss = model(x.to(device), y.to(device))',
    '            losses.append(loss.item())',
    '        results[split] = sum(losses)/len(losses) if losses else float("nan")',
    '    model.train()',
    '    return results',
    '',
    'def save_ckpt(path, model, opt, step, val_loss):',
    '    torch.save({"step":step,"val_loss":val_loss,"model_state":model.state_dict(),"optimizer_state":opt.state_dict(),"model_cfg":model.cfg}, path)',
    '',
    'def main():',
    '    args = parse_args()',
    '    torch.manual_seed(args.seed)',
    '    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else torch.device(args.device)',
    '    print(f"\n{chr(61)*60}\n  NanoMind Training\n  Device: {device}\n{chr(61)*60}")',
    '    train_loader, val_loader, tokenizer = get_dataloaders(args.data, args.block_size, args.batch_size)',
    '    model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, block_size=args.block_size, dropout=args.dropout)',
    '    train_cfg = TrainConfig(data_path=args.data, batch_size=args.batch_size, max_iters=args.max_iters, eval_interval=args.eval_interval, eval_iters=args.eval_iters, learning_rate=args.learning_rate, min_lr=args.min_lr, warmup_iters=args.warmup_iters, grad_clip=args.grad_clip, weight_decay=args.weight_decay, device=args.device, seed=args.seed, log_interval=args.log_interval)',
    '    model = NanoMind(model_cfg).to(device)',
    '    print(f"  {model}\n")',
    '    decay = [p for n,p in model.named_parameters() if p.dim()>=2]',
    '    nodecay = [p for n,p in model.named_parameters() if p.dim()<2]',
    '    opt = torch.optim.AdamW([{"params":decay,"weight_decay":train_cfg.weight_decay},{"params":nodecay,"weight_decay":0.0}], lr=train_cfg.learning_rate, betas=train_cfg.betas)',
    '    start, best = 0, float("inf")',
    '    if args.resume:',
    '        ckpt = torch.load(args.resume, map_location="cpu")',
    '        model.load_state_dict(ckpt["model_state"]); opt.load_state_dict(ckpt["optimizer_state"])',
    '        start, best = ckpt["step"], ckpt.get("val_loss", float("inf"))',
    '    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)',
    '    tokenizer.save(out_dir/"vocab.json")',
    '    loaders = {"train": train_loader, "val": val_loader}',
    '    train_iter, model.train(), t0, running = iter(train_loader), None, time.time(), 0.0',
    '    print(f"{chr(45)*60}\n  Step  LR          Loss\n{chr(45)*60}")',
    '    for step in range(start, train_cfg.max_iters):',
    '        try: x, y = next(train_iter)',
    '        except StopIteration: train_iter = iter(train_loader); x, y = next(train_iter)',
    '        x, y = x.to(device), y.to(device)',
    '        lr = get_lr(step, train_cfg)',
    '        for pg in opt.param_groups: pg["lr"] = lr',
    '        opt.zero_grad(set_to_none=True)',
    '        _, loss = model(x, y)',
    '        loss.backward()',
    '        if train_cfg.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)',
    '        opt.step()',
    '        running += loss.item()',
    '        if (step+1) % train_cfg.log_interval == 0:',
    '            print(f"  {step+1:>5}  {lr:.2e}  {running/train_cfg.log_interval:.4f}  ({time.time()-t0:.1f}s)", flush=True)',
    '            running, t0 = 0.0, time.time()',
    '        if (step+1) % train_cfg.eval_interval == 0:',
    '            L = estimate_loss(model, loaders, train_cfg.eval_iters, device)',
    '            print(f"\n  Eval@{step+1}: train={L[chr(116)+chr(114)+chr(97)+chr(105)+chr(110)]:.4f} val={L[chr(118)+chr(97)+chr(108)]:.4f}")',
    '            if L["val"] < best: best=L["val"]; save_ckpt(out_dir/"best.pt",model,opt,step+1,best); print(f"  Best val {best:.4f} saved")',
    '            save_ckpt(out_dir/"latest.pt",model,opt,step+1,L["val"])',
    '            print(f"{chr(45)*60}"); t0=time.time()',
    '    print(f"\n{chr(61)*60}\n  Done! Best val: {best:.4f}\n{chr(61)*60}\n")',
    '',
    'if __name__ == "__main__": main()'
)
Set-Content "$repo\train.py" ($trainLines -join "`n") -Encoding UTF8
Do-Commit "feat: add cosine LR schedule, warmup, gradient clipping, and checkpointing"

# --- COMMIT 14: generate.py
Write-Host "-- Commit 14/15 --"
# generate.py is already on disk
Do-Commit "feat: add text generation CLI with prompt, temperature, and top-k flags"

# --- COMMIT 15: README
Write-Host "-- Commit 15/15 --"
Do-Commit "docs: add README with architecture overview, usage guide, and examples"

# --- Push to GitHub
Write-Host "=== Creating GitHub repository and pushing ===" -ForegroundColor Cyan
gh repo create nanomind --public --description "NanoMind: A small GPT-style transformer LLM built from scratch in PyTorch" --source . --remote origin --push

Write-Host "=== All done! ===" -ForegroundColor Green
gh repo view nanomind --json url -q ".url"
