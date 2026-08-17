# Changelog

All notable changes to NanoMind are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Project scaffold, tooling, and CI pipeline (Day 1)
- Character-level tokenizer with BOS/EOS/PAD/UNK, save/load, factory (Day 2)
- BPE tokenizer with merge learning, encode/decode, persistence, factory (Day 3)
- Data pipeline: TextDataset, IterableTextDataset, DataLoaders, PrefetchLoader, stats (Day 4)
- Attention: CausalSelfAttention, KVCache, causal mask, Flash Attention dispatch (Day 5)
- Blocks: TransformerBlock (Pre/Post-LN), FeedForward (GELU/SwiGLU), RMSNorm, LayerNorm (Day 6)
- Full model: NanoMind with embeddings, N blocks, weight tying, generate(), ModelConfig (Day 7)
- Training: Trainer loop with AMP, grad accum, gradient clip, early stop, estimate_loss (Day 8)
- BPE tokenizer with merge learning, encode/decode, persistence, factory (Day 3)
- Data pipeline: TextDataset, IterableTextDataset, DataLoaders, PrefetchLoader, stats (Day 4)
- Attention: CausalSelfAttention, KVCache, causal mask, Flash Attention dispatch (Day 5)
- Blocks: TransformerBlock (Pre/Post-LN), FeedForward (GELU/SwiGLU), RMSNorm, LayerNorm (Day 6)
- Full model: NanoMind with embeddings, N blocks, weight tying, generate(), ModelConfig (Day 7)
- Training: Trainer loop with AMP, grad accum, gradient clip, early stop, estimate_loss (Day 8)
- Coloured logging utility (`nanomind.utils.logger`)
- Reproducibility utilities (`nanomind.utils.seed`)
- Device detection (`nanomind.utils.device`)
- Benchmarking timer (`nanomind.utils.timer`)
