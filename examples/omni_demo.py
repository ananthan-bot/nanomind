"""
examples/omni_demo.py — End-to-end Demonstration of Unified Omni-Modal Foundation Model.
Demonstrates:
  1. Multimodal Token Interleaving (Text + Vision + Audio)
  2. NanoMind-Omni Unified Forward Pass & Dual-Head Output
  3. Full-Duplex Streaming Dialogue & Barge-in Interruption Detection
  4. Voice Activity Detection (VAD) & End-of-Utterance Handling
  5. Time-to-First-Audio (TTFA) Latency Benchmarking
"""
import torch
from nanomind.omni import (
    OmniConfig,
    NanoMindOmni,
    OmniPipeline,
    MultimodalSequenceBuilder,
    ModalityType,
    DuplexDialogueManager,
    VoiceActivityDetector,
    EndOfUtterancePredictor,
    OmniBenchmark,
)


def run_demo():
    print("=" * 70)
    print("  NanoMind Day 60: 💎 DIAMOND JUBILEE — Omni-Modal Foundation Model")
    print("=" * 70)

    # 1. Multimodal Sequence Interleaving
    print("\n[1] Multimodal Sequence Interleaving:")
    builder = MultimodalSequenceBuilder(d_model=64)
    text_embeds = torch.randn(1, 5, 64)   # 5 text tokens
    image_embeds = torch.randn(1, 4, 64)  # 4 vision patches
    audio_embeds = torch.randn(1, 6, 64)  # 6 audio frames

    seq, mod_ids = builder.build_sequence([
        (ModalityType.TEXT, text_embeds),
        (ModalityType.IMAGE, image_embeds),
        (ModalityType.AUDIO, audio_embeds),
    ])
    print(f"  Combined Sequence shape : {seq.shape} (T=15 total multimodal tokens)")
    print(f"  Modality IDs sequence   : {mod_ids.tolist()[0]} (0=text, 1=img, 2=audio)")

    # 2. Unified NanoMindOmni Model Forward Pass
    print("\n[2] NanoMind-Omni Unified Forward Pass:")
    cfg = OmniConfig(d_model=64, n_layers=2, n_heads=4, text_vocab_size=1000, audio_codebook_size=512)
    model = NanoMindOmni(cfg)

    text_input = torch.randint(0, 1000, (1, 4))
    img_patches = torch.randn(1, 4, 256)
    audio_mel = torch.randn(1, 8, 80)

    outputs = model(
        text_tokens=text_input,
        image_patches=img_patches,
        audio_frames=audio_mel,
    )
    print(f"  Hidden representations shape : {outputs['hidden_states'].shape}")
    print(f"  Text Logits shape            : {outputs['text_logits'].shape}")
    print(f"  Speech RVQ Logits shape      : {outputs['speech_logits'].shape}")
    print(f"  Tool Trigger Logits shape    : {outputs['tool_logits'].shape}")

    # 3. High-Level OmniPipeline
    print("\n[3] High-Level OmniPipeline Chat:")
    pipeline = OmniPipeline(model)
    resp = pipeline.chat(text_tokens=text_input, image_patches=img_patches)
    print(f"  Generated Text Token ID : {resp.text_token_ids[0]}")
    print(f"  Generated Speech Code   : {resp.speech_token_ids[0]}")
    print(f"  Tool Call Triggered?    : {resp.has_tool_trigger}")

    # 4. Duplex Dialogue & Barge-in Interruption Detection
    print("\n[4] Full-Duplex Streaming & Interruption:")
    duplex = DuplexDialogueManager(barge_in_sensitivity=0.6)
    duplex.start_speaking()
    print(f"  State before interruption: {duplex.state.value}")

    # Simulate user speaking loudly (VAD confidence = 0.85) while model is speaking
    step_info = duplex.on_user_speech_frame(user_speaking_prob=0.85)
    print(f"  User spoke with VAD=0.85 -> Interrupted: {step_info['interrupted']}")
    print(f"  New State after barge-in : {step_info['current_state']}")

    # 5. Voice Activity Detection & Turn-Taking
    print("\n[5] Voice Activity Detection (VAD) & End-of-Utterance:")
    vad = VoiceActivityDetector(energy_threshold=0.02)
    silence_chunk = torch.zeros(1600)  # Silence
    speech_chunk = torch.randn(1600) * 0.1  # Active speech
    print(f"  Silent frame VAD: {vad.is_speech(silence_chunk)['is_speech']}")
    print(f"  Speech frame VAD: {vad.is_speech(speech_chunk)['is_speech']}")

    # 6. Latency Benchmark
    print("\n[6] Time-to-First-Audio (TTFA) Benchmark:")
    bm = OmniBenchmark()
    lat = bm.measure_generation_latency(lambda: model(text_tokens=text_input), iterations=3)
    print(f"  Mean Generation Latency: {lat['mean_latency_ms']} ms")
    print(f"  Real-time conversational ready (<300ms): {lat['real_time_dialogue_ready']}")

    print("\n[OK] 💎 NanoMind Diamond Jubilee Omni Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
