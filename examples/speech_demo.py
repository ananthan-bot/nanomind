"""
examples/speech_demo.py — End-to-end Demonstration of Speech & Audio-Language Models.
Demonstrates:
  1. Mel Spectrogram Extraction from Audio Waveform
  2. Residual Vector Quantization (RVQ) Codec Encoding & Decoding
  3. Whisper-style Audio Transformer Encoder Forward Pass
  4. End-to-End SpeechLanguageModel Multimodal Output
  5. Streaming Audio Buffer & Real-Time Factor (RTF) Profiling
  6. Word Error Rate (WER) Evaluation
"""
import time
import torch

from nanomind.speech import (
    AudioConfig,
    MelSpectrogramExtractor,
    CodecConfig,
    ResidualVectorQuantizer,
    SpeechEncoderConfig,
    AudioEncoder,
    SpeechLMConfig,
    SpeechLanguageModel,
    StreamingAudioBuffer,
    RTFProfiler,
    word_error_rate,
    compute_codebook_perplexity,
)


def run_demo():
    print("=" * 70)
    print("  NanoMind Day 58: Speech & Audio-Language Models (Whisper / RVQ)")
    print("=" * 70)

    # 1. Audio Waveform & Mel-Spectrogram Extraction
    print("\n[1] Mel-Spectrogram Extraction:")
    sr = 16000
    duration_sec = 1.0
    t = torch.linspace(0, duration_sec, int(sr * duration_sec))
    # Synthetic audio signal with harmonics (440 Hz + 880 Hz)
    waveform = 0.5 * torch.sin(2 * 3.14159 * 440 * t) + 0.3 * torch.sin(2 * 3.14159 * 880 * t)
    waveform = waveform.unsqueeze(0)  # (1, 16000)

    extractor = MelSpectrogramExtractor(AudioConfig(sample_rate=sr, n_mels=80))
    log_mel = extractor(waveform)
    print(f"  Raw Waveform shape: {waveform.shape} (1 second at {sr} Hz)")
    print(f"  Log-Mel Spectrogram shape: {log_mel.shape} (80 mel bands x {log_mel.shape[2]} frames)")

    # 2. Residual Vector Quantization (RVQ)
    print("\n[2] Residual Vector Quantization (RVQ) 4-Stage Codec:")
    codec_cfg = CodecConfig(codebook_size=512, n_q=4, embedding_dim=64)
    rvq = ResidualVectorQuantizer(codec_cfg)

    # Continuous audio latents (B=1, D=64, T=50)
    z = torch.randn(1, 64, 50)
    z_q, loss_rvq, codes = rvq(z)
    recon = rvq.decode(codes)
    perplexity = compute_codebook_perplexity(codes, codebook_size=512)

    print(f"  Input Latent shape: {z.shape}")
    print(f"  RVQ Codes shape: {codes.shape} (4 quantization stages)")
    print(f"  Reconstructed shape: {recon.shape}")
    print(f"  Quantization Commitment Loss: {loss_rvq.item():.4f}")
    print(f"  Codebook Perplexity: {perplexity:.1f} / 512")

    # 3. Whisper-Style Audio Transformer Encoder
    print("\n[3] Whisper-Style Audio Encoder:")
    enc_cfg = SpeechEncoderConfig(n_mels=80, d_model=128, n_layers=2, n_heads=4)
    encoder = AudioEncoder(enc_cfg)
    audio_feats = encoder(log_mel)
    print(f"  Downsampled Audio Features: {audio_feats.shape} (4x temporal reduction)")

    # 4. End-to-End SpeechLanguageModel
    print("\n[4] End-to-End Speech-Language Model (SpeechLM):")
    lm_cfg = SpeechLMConfig(
        llm_d_model=128,
        text_vocab_size=1000,
        audio_vocab_size=512,
    )
    model = SpeechLanguageModel(lm_cfg)
    text_prompt = torch.randint(0, 1000, (1, 8))
    out = model(waveform=waveform, text_tokens=text_prompt)

    print(f"  Text Logits shape: {out['text_logits'].shape}")
    print(f"  Audio Logits shape: {out['audio_logits'].shape}")

    # 5. Real-Time Streaming Audio Buffer & RTF Profiler
    print("\n[5] Streaming Audio & RTF Profiler:")
    stream_buf = StreamingAudioBuffer(sample_rate=16000, chunk_ms=200)
    profiler = RTFProfiler()

    # Simulate 5 consecutive 200ms audio chunks
    chunk_samples = int(16000 * 0.2)
    for _ in range(5):
        audio_chunk = torch.randn(chunk_samples)
        stream_buf.append(audio_chunk)

        t0 = time.time()
        c = stream_buf.get_next_chunk()
        proc_time = time.time() - t0 + 0.01  # Simulated ~10ms forward pass
        profiler.record_chunk(audio_duration_sec=0.2, processing_time_sec=proc_time)

    prof_summary = profiler.summary()
    print(f"  Total Processed Audio: {prof_summary['total_audio_sec']} s")
    print(f"  Average Chunk Latency: {prof_summary['avg_latency_ms']} ms")
    print(f"  Real-Time Factor (RTF): {prof_summary['rtf']} (< 1.0 means faster than real-time)")

    # 6. Word Error Rate (WER)
    print("\n[6] Speech Recognition Evaluation:")
    ref = "the quick brown fox jumps over the lazy dog"
    hyp = "the quick brown fox jumped over a lazy dog"
    wer = word_error_rate(ref, hyp)
    print(f"  Reference : '{ref}'")
    print(f"  Hypothesis: '{hyp}'")
    print(f"  Word Error Rate (WER): {wer * 100:.1f}%")

    print("\n[OK] All Speech & Audio-Language Model demos completed successfully!")


if __name__ == "__main__":
    run_demo()
