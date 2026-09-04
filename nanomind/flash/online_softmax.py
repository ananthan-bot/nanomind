"""
nanomind/flash/online_softmax.py — Online (streaming) softmax for Flash Attention.

Standard softmax requires two passes over the data:
  Pass 1: find max(x_i) for numerical stability
  Pass 2: compute exp(x_i - max) / Σ exp(x_j - max)

Online softmax (Milakov & Gimelshein, 2018) merges both passes into one,
keeping running statistics (max and sum) that can be updated incrementally
as new tiles of K/V are processed.

This is the mathematical heart of Flash Attention: it lets us update the
running output accumulator O and the normalization constant l with each
new tile, without ever needing the full softmax row at once.

State after processing tile t:
  m_t = max(s_1 ... s_t)   — running max of scores seen so far
  l_t = Σ exp(s_i - m_t)   — running normalisation denominator
  O_t = output accumulator

When a new tile t+1 arrives with block scores S_{t+1}:
  m_{t+1} = max(m_t, max(S_{t+1}))
  l_{t+1} = l_t × exp(m_t - m_{t+1}) + Σ exp(S_{t+1,j} - m_{t+1})
  O_{t+1} = (O_t × l_t × exp(m_t - m_{t+1}) + exp(S_{t+1}) × V_{t+1}) / l_{t+1}
"""

from __future__ import annotations

import torch


class OnlineSoftmaxState:
    """
    Running state for online (streaming) softmax accumulation.

    Holds the running max ``m``, normalisation sum ``l``, and
    output accumulator ``O`` for a batch of query rows.

    Args:
        q_block: Query block ``(B, H, Bq, Dh)`` used to infer shape and device.
    """

    __slots__ = ("m", "l", "O")

    def __init__(self, q_block: torch.Tensor) -> None:
        B, H, Bq, Dh = q_block.shape
        device, dtype = q_block.device, q_block.dtype
        self.m = torch.full((B, H, Bq, 1), float("-inf"), device=device, dtype=dtype)
        self.l = torch.zeros((B, H, Bq, 1), device=device, dtype=dtype)
        self.O = torch.zeros((B, H, Bq, Dh), device=device, dtype=dtype)

    def update(
        self,
        s_block: torch.Tensor,   # (B, H, Bq, Bkv) raw scores for this KV tile
        v_block: torch.Tensor,   # (B, H, Bkv, Dh)
    ) -> None:
        """
        Incorporate a new KV tile into the running output.

        Args:
            s_block: Attention scores for this tile ``(B, H, Bq, Bkv)``.
            v_block: Value tile ``(B, H, Bkv, Dh)``.
        """
        # New running max
        m_new = torch.maximum(self.m, s_block.max(dim=-1, keepdim=True).values)

        # Rescale existing accumulator and l
        scale_old = torch.exp(self.m - m_new)
        exp_s     = torch.exp(s_block - m_new)

        l_new = self.l * scale_old + exp_s.sum(dim=-1, keepdim=True)
        O_new = self.O * scale_old + torch.matmul(exp_s, v_block)

        self.m = m_new
        self.l = l_new
        self.O = O_new

    def finalize(self) -> torch.Tensor:
        """
        Return the final normalised attention output.

        Returns:
            ``(B, H, Bq, Dh)`` output tensor.
        """
        return self.O / (self.l + 1e-8)
