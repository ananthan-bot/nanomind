"""
nanomind/omni/model.py — NanoMindOmni: Unified Omni-Modal Foundation Model.
Unifies Text + Vision Patches + Audio Mel Frames into a shared Transformer backbone.
"""
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomind.omni.config import OmniConfig, ModalityType
from nanomind.omni.fusion import ModalityEmbedding, CrossModalFusionLayer
from nanomind.omni.generator import OmniGenerator


class NanoMindOmni(nn.Module):
    """
    NanoMind-Omni unified foundation model architecture.
    Processes any combination of text, images, and audio, and outputs text and speech.
    """

    def __init__(self, config: Optional[OmniConfig] = None):
        super().__init__()
        self.config = config or OmniConfig()
        d_model = self.config.d_model

        # 1. Modality Input Encoders / Linear Projectors
        self.text_embed = nn.Embedding(self.config.text_vocab_size, d_model)
        self.image_projector = nn.Linear(self.config.image_patch_dim, d_model)
        self.audio_projector = nn.Linear(self.config.audio_mel_dim, d_model)

        # 2. Modality Type Embedding
        self.modality_embed = ModalityEmbedding(num_modalities=4, d_model=d_model)

        # 3. Position Encodings
        self.pos_embed = nn.Parameter(torch.randn(1, self.config.max_seq_len, d_model) * 0.02)

        # 4. Cross-Modal Transformer Backbone
        self.blocks = nn.ModuleList([
            CrossModalFusionLayer(
                d_model=d_model,
                n_heads=self.config.n_heads,
                dim_feedforward=self.config.dim_feedforward,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # 5. OmniGenerator Heads
        self.generator = OmniGenerator(self.config)

    def forward(
        self,
        text_tokens: Optional[torch.Tensor] = None,
        image_patches: Optional[torch.Tensor] = None,
        audio_frames: Optional[torch.Tensor] = None,
        modality_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Multimodal forward pass.
        text_tokens: (B, T_text)
        image_patches: (B, T_img, image_patch_dim)
        audio_frames: (B, T_audio, audio_mel_dim)
        """
        tokens_list = []
        mod_ids_list = []

        if text_tokens is not None:
            B, T_t = text_tokens.shape
            h_text = self.text_embed(text_tokens)
            tokens_list.append(h_text)
            mod_ids_list.append(torch.zeros(B, T_t, dtype=torch.long, device=text_tokens.device))

        if image_patches is not None:
            B, T_i, _ = image_patches.shape
            h_img = self.image_projector(image_patches)
            tokens_list.append(h_img)
            mod_ids_list.append(torch.ones(B, T_i, dtype=torch.long, device=image_patches.device))

        if audio_frames is not None:
            B, T_a, _ = audio_frames.shape
            h_audio = self.audio_projector(audio_frames)
            tokens_list.append(h_audio)
            mod_ids_list.append(torch.full((B, T_a), 2, dtype=torch.long, device=audio_frames.device))

        if not tokens_list:
            raise ValueError("At least one input modality must be provided")

        x = torch.cat(tokens_list, dim=1)
        B, T_total, D = x.shape

        if modality_ids is None:
            mod_ids = torch.cat(mod_ids_list, dim=1)
        else:
            mod_ids = modality_ids

        # Add modality embedding + positional encoding
        x = x + self.modality_embed(mod_ids)
        x = x + self.pos_embed[:, :T_total, :]

        # Backbone transformer passes
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)

        # Multi-head generation
        logits_dict = self.generator(h)
        logits_dict["hidden_states"] = h
        return logits_dict
