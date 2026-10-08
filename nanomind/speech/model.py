"""
nanomind/speech/model.py — End-to-End Speech-Language Model (Mini-Omni / SpeechGPT style).
Jointly processes spoken audio inputs and generates text and acoustic RVQ tokens.
"""
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn

from nanomind.speech.config import SpeechLMConfig
from nanomind.speech.features import MelSpectrogramExtractor
from nanomind.speech.encoder import AudioEncoder
from nanomind.speech.projector import SpeechProjector
from nanomind.speech.codec import ResidualVectorQuantizer


class SpeechLanguageModel(nn.Module):
    """
    Multimodal Speech-Language Model:
    1. Audio feature extraction (MelSpectrogramExtractor)
    2. Audio encoding (AudioEncoder)
    3. Multimodal projection (SpeechProjector)
    4. Text embedding + Backbone LLM processing
    5. Dual prediction heads:
       - Text head: predicts next textual token
       - Codec head: predicts next acoustic RVQ tokens
    """

    def __init__(self, config: Optional[SpeechLMConfig] = None):
        super().__init__()
        self.config = config or SpeechLMConfig()

        # Audio front-end
        self.mel_extractor = MelSpectrogramExtractor(self.config.audio)
        self.audio_encoder = AudioEncoder(self.config.encoder)
        self.projector = SpeechProjector(
            audio_dim=self.config.encoder.d_model,
            llm_dim=self.config.llm_d_model,
            downsample_rate=self.config.projector_downsample_rate,
        )

        # RVQ neural audio codec for speech synthesis
        self.codec = ResidualVectorQuantizer(self.config.codec)

        # Text embedding
        self.text_embedding = nn.Embedding(self.config.text_vocab_size, self.config.llm_d_model)

        # Backbone transformer layers
        self.llm_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.config.llm_d_model,
                nhead=4,
                dim_feedforward=4 * self.config.llm_d_model,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(2)
        ])
        self.ln_f = nn.LayerNorm(self.config.llm_d_model)

        # Output heads
        self.text_head = nn.Linear(self.config.llm_d_model, self.config.text_vocab_size, bias=False)
        self.audio_head = nn.Linear(self.config.llm_d_model, self.config.audio_vocab_size, bias=False)

    def encode_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Raw audio -> Mel -> Encoder -> Projector -> LLM-compatible audio tokens."""
        mel = self.mel_extractor(waveform)
        enc_out = self.audio_encoder(mel)
        audio_tokens = self.projector(enc_out)
        return audio_tokens

    def forward(
        self,
        waveform: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Multimodal forward pass.
        waveform: (B, T_audio_samples)
        text_tokens: (B, T_text)
        Returns:
            dict containing:
              - 'text_logits': (B, T_total, text_vocab_size)
              - 'audio_logits': (B, T_total, audio_vocab_size)
        """
        prefix_embeds = []

        if waveform is not None:
            audio_embeds = self.encode_audio(waveform)
            prefix_embeds.append(audio_embeds)

        if text_tokens is not None:
            text_embeds = self.text_embedding(text_tokens)
            prefix_embeds.append(text_embeds)

        if not prefix_embeds:
            raise ValueError("Must provide either waveform or text_tokens")

        # Concatenate audio + text embeddings along sequence dimension
        x = torch.cat(prefix_embeds, dim=1)

        # LLM forward
        for block in self.llm_blocks:
            x = block(x)
        h = self.ln_f(x)

        text_logits = self.text_head(h)
        audio_logits = self.audio_head(h)

        return {
            "hidden_states": h,
            "text_logits": text_logits,
            "audio_logits": audio_logits,
        }
