"""
nanomind/moe/model.py — NanoMind with Mixture of Experts FFN layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanomind.model.config import ModelConfig
from nanomind.moe.config import MoEConfig
from nanomind.moe.block import MoETransformerBlock
from nanomind.norm.factory import get_norm
from nanomind.utils.logger import get_logger

log = get_logger("moe.model")


class NanoMindMoE(nn.Module):
    """
    NanoMind transformer model with Sparse Mixture of Experts FFN layers.

    Every transformer block uses a SparseMoELayer instead of the dense FFN.
    The forward pass returns both logits and the summed auxiliary load-balancing
    loss (to be added to the cross-entropy loss during training).

    Args:
        model_cfg: Standard model configuration.
        moe_cfg:   MoE configuration.

    Example::

        cfg     = ModelConfig(vocab_size=32000, block_size=512, d_model=256,
                              n_layers=6, n_heads=8)
        moe_cfg = MoEConfig(num_experts=8, top_k=2)
        model   = NanoMindMoE(cfg, moe_cfg)

        logits, aux_loss = model(input_ids)
        loss = cross_entropy(logits, labels) + aux_loss
    """

    def __init__(self, model_cfg: ModelConfig, moe_cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg     = model_cfg
        self.moe_cfg = moe_cfg

        self.tok_emb = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        # Learned pos embedding (MoE typically uses RoPE, but keep it flexible)
        if model_cfg.pos_type in ("learned", None, ""):
            self.pos_emb = nn.Embedding(model_cfg.block_size, model_cfg.d_model)
        else:
            self.pos_emb = None

        self.drop    = nn.Dropout(model_cfg.dropout)
        self.blocks  = nn.ModuleList([
            MoETransformerBlock(
                d_model=model_cfg.d_model,
                n_heads=model_cfg.n_heads,
                block_size=model_cfg.block_size,
                moe_cfg=moe_cfg,
                dropout=model_cfg.dropout,
                norm_type=getattr(model_cfg, "norm_type", "layernorm"),
                pos_type=model_cfg.pos_type,
                n_kv_heads=model_cfg.n_kv_heads,
                window_size=model_cfg.window_size,
            )
            for _ in range(model_cfg.n_layers)
        ])
        self.norm    = get_norm(getattr(model_cfg, "norm_type", "layernorm"), model_cfg.d_model)
        self.lm_head = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        log.info(f"NanoMindMoE: {n_params:,} params | "
                 f"{moe_cfg.num_experts} experts, top-{moe_cfg.top_k}")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        idx:    torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            idx:     Token IDs ``(B, T)``.
            targets: Target token IDs for loss computation ``(B, T)``.

        Returns:
            Tuple of:
            - Logits ``(B, T, vocab_size)``
            - Combined loss (cross-entropy + aux) if targets given, else aux_loss total
        """
        B, T = idx.shape
        x    = self.tok_emb(idx)

        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x   = x + self.pos_emb(pos)

        x = self.drop(x)

        total_aux = torch.tensor(0.0, device=idx.device)
        for block in self.blocks:
            x, aux = block(x)
            total_aux = total_aux + aux

        x      = self.norm(x)
        logits = self.lm_head(x)

        if targets is not None:
            import torch.nn.functional as F
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            return logits, loss + total_aux

        return logits, total_aux

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_active_parameters(self) -> int:
        """
        Compute the number of parameters *active* per forward token.

        Unlike ``num_parameters()`` which counts all weights including
        the unused expert FFNs, this returns the effective parameter count
        that actually contributes to each token's computation:
        non-MoE params + top_k/num_experts × MoE params.
        """
        from nanomind.moe.layer import SparseMoELayer
        total_moe, total_non_moe = 0, 0
        for name, module in self.named_modules():
            if isinstance(module, SparseMoELayer):
                # Expert params: each expert is equally sized
                expert_params = sum(
                    p.numel() for e in module.experts for p in e.parameters()
                )
                router_params = sum(p.numel() for p in module.router.parameters())
                # Only top_k experts activate per token
                active_expert = int(expert_params * self.moe_cfg.top_k / self.moe_cfg.num_experts)
                total_moe    += active_expert + router_params
            elif not any(isinstance(module, SparseMoELayer)
                         for module in module.modules()):
                pass
        non_moe = sum(
            p.numel() for n, p in self.named_parameters()
            if 'experts.' not in n
        )
        return non_moe + total_moe

    def __repr__(self) -> str:
        return (
            f"NanoMindMoE("
            f"params={self.num_parameters():,}, "
            f"experts={self.moe_cfg.num_experts}, "
            f"top_k={self.moe_cfg.top_k})"
        )
