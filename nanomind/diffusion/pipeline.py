"""
nanomind/diffusion/pipeline.py — End-to-end Diffusion LM pipeline.

Unified interface combining:
  - NoiseSchedule
  - DiffusionDenoiser
  - DDPMTrainer / DDIMSampler
  - Masked diffusion option
"""

from __future__ import annotations
import torch
import torch.optim as optim

from nanomind.diffusion.schedule import NoiseSchedule
from nanomind.diffusion.denoiser import DiffusionDenoiser
from nanomind.diffusion.ddpm import DDPMTrainer, DDPMSampler
from nanomind.diffusion.ddim import DDIMSampler
from nanomind.diffusion.masked import MaskedDiffusionLM
from nanomind.utils.logger import get_logger

log = get_logger("diffusion.pipeline")


class DiffusionLMPipeline:
    """
    End-to-end Diffusion Language Model pipeline.

    Supports two modes:
      - ``"continuous"``: Gaussian diffusion in embedding space (DDPM/DDIM)
      - ``"masked"``:     Token masking diffusion (D3PM/MDLM)

    Args:
        vocab_size: Vocabulary size.
        d_model:    Embedding dimension.
        n_layers:   Transformer layers.
        n_heads:    Attention heads.
        max_seq:    Maximum sequence length.
        n_steps:    Diffusion steps.
        schedule:   Noise schedule type.
        mode:       ``"continuous"`` or ``"masked"``.

    Example::

        pipe = DiffusionLMPipeline(vocab_size=100, d_model=64, mode="continuous")
        pipe.train_step(token_ids)
        tokens = pipe.generate(batch_size=2, seq_len=16, n_steps=20)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int   = 64,
        n_layers:   int   = 3,
        n_heads:    int   = 4,
        max_seq:    int   = 32,
        n_steps:    int   = 100,
        schedule:   str   = "cosine",
        mode:       str   = "continuous",
        lr:         float = 1e-3,
    ) -> None:
        assert mode in ("continuous", "masked")
        self.mode    = mode
        self.n_steps = n_steps

        if mode == "continuous":
            self.noise_schedule = NoiseSchedule(n_steps=n_steps, schedule=schedule)
            self.denoiser       = DiffusionDenoiser(
                vocab_size, d_model, n_layers, n_heads, max_seq
            )
            self._trainer = DDPMTrainer(self.denoiser, self.noise_schedule)
            self._ddpm    = DDPMSampler(self.denoiser, self.noise_schedule)
            self._ddim    = DDIMSampler(self.denoiser, self.noise_schedule, eta=0.0)
            self.opt      = optim.Adam(self.denoiser.parameters(), lr=lr)
        else:
            self.denoiser = MaskedDiffusionLM(
                vocab_size, d_model, n_layers, n_heads, max_seq
            )
            self.opt      = optim.Adam(self.denoiser.parameters(), lr=lr)

    def train_step(
        self,
        token_ids: torch.Tensor,
        loss_type: str = "simple",
    ) -> float:
        """
        Single training step.

        Args:
            token_ids: ``(B, T)`` token IDs.
            loss_type: ``"simple"`` or ``"vlb"`` (continuous mode only).

        Returns:
            Loss value.
        """
        self.opt.zero_grad()
        if self.mode == "continuous":
            loss = self._trainer.loss(token_ids, loss_type=loss_type)
        else:
            t    = torch.randint(1, self.n_steps + 1, (1,)).item()
            loss = self.denoiser.loss(token_ids, t=t, T=self.n_steps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        seq_len:    int = 16,
        n_steps:    int = 20,
        sampler:    str = "ddim",
    ) -> torch.Tensor:
        """
        Generate token sequences.

        Args:
            batch_size: Number of sequences.
            seq_len:    Sequence length.
            n_steps:    Sampling steps.
            sampler:    ``"ddpm"``, ``"ddim"``, or ``"masked"`` (auto).

        Returns:
            ``(B, T)`` token IDs.
        """
        if self.mode == "masked":
            return self.denoiser.generate(batch_size, seq_len, T=n_steps)
        elif sampler == "ddim":
            return self._ddim.sample(batch_size, seq_len, n_steps=n_steps)
        else:
            return self._ddpm.sample(batch_size, seq_len, n_steps=n_steps)

    def info(self) -> dict:
        """Return pipeline configuration summary."""
        n_params = sum(p.numel() for p in self.denoiser.parameters())
        d = {
            "mode":     self.mode,
            "n_steps":  self.n_steps,
            "n_params": n_params,
        }
        if self.mode == "continuous":
            d["schedule"] = self.noise_schedule.to_dict()
        return d
