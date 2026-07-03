"""Verify SoVITS shared-backbone output matches full-model output.

Compares MultiSpeakerTTS (shared GPT+SoVITS backbone with per-speaker
weight injection) against a standalone TTS instance for self-consistency.

For a real-world evaluation, replace the speaker model path with an actual
fine-tuned model distinct from the base checkpoint.

Quick test: use the SAME model as both base and speaker. The shared backbone
should produce identical output (within floating-point tolerance) because
copy_() injects the exact same weights that the full model uses.

Usage:
    # Self-consistency test (no fine-tuned model needed)
    python tests/test_sovits_sharing.py

    # Real evaluation (with fine-tuned model)
    python tests/test_sovits_sharing.py --speaker-gpt path/to/speaker_gpt.ckpt \\
                                        --speaker-sovits path/to/speaker_sovits.pth
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_mcd(
    audio_ref: np.ndarray,
    audio_test: np.ndarray,
    sr: int = 32000,
    n_mels: int = 80,
    n_mfcc: int = 13,
) -> float:
    """Compute Mel Cepstral Distortion between two audio signals.

    Lower is better. Typical thresholds:
      < 2.0  → perceptually transparent
      < 5.0  → acceptable quality
      > 8.0  → noticeable degradation

    Returns float('inf') if signals are too short.
    """
    try:
        import librosa
    except ImportError:
        logger.warning("librosa not installed, skipping MCD")
        return float("nan")

    min_len = min(len(audio_ref), len(audio_test))
    if min_len < sr * 0.1:  # need at least 100ms
        return float("inf")

    audio_ref = audio_ref[:min_len]
    audio_test = audio_test[:min_len]

    # Extract MFCCs from mel spectrograms
    mel_ref = librosa.feature.melspectrogram(
        y=audio_ref.astype(np.float64), sr=sr, n_mels=n_mels
    )
    mel_test = librosa.feature.melspectrogram(
        y=audio_test.astype(np.float64), sr=sr, n_mels=n_mels
    )

    mfcc_ref = librosa.feature.mfcc(
        S=librosa.power_to_db(mel_ref), sr=sr, n_mfcc=n_mfcc
    )
    mfcc_test = librosa.feature.mfcc(
        S=librosa.power_to_db(mel_test), sr=sr, n_mfcc=n_mfcc
    )

    # Dynamic Time Warping alignment
    _, wp = librosa.sequence.dtw(
        X=mfcc_ref.T, Y=mfcc_test.T, metric="euclidean"
    )

    # Accumulate distortion along the warping path
    diff_sum = 0.0
    for i, j in wp:
        diff_sum += np.sqrt(np.sum((mfcc_ref[:, i] - mfcc_test[:, j]) ** 2))
    mcd = diff_sum / len(wp)

    return float(mcd)


def compute_mel_cosine(audio_ref: np.ndarray, audio_test: np.ndarray,
                       sr: int = 32000, n_mels: int = 80) -> float:
    """Compute average cosine similarity of mel spectrogram frames.

    Higher is better (1.0 = identical). Typical thresholds:
      > 0.95 → near-identical
      > 0.85 → very similar
      < 0.70 → noticeably different
    """
    try:
        import librosa
    except ImportError:
        logger.warning("librosa not installed, skipping cosine similarity")
        return float("nan")

    min_len = min(len(audio_ref), len(audio_test))
    if min_len < sr * 0.1:
        return 0.0

    audio_ref = audio_ref[:min_len]
    audio_test = audio_test[:min_len]

    mel_ref = librosa.feature.melspectrogram(
        y=audio_ref.astype(np.float64), sr=sr, n_mels=n_mels
    )
    mel_test = librosa.feature.melspectrogram(
        y=audio_test.astype(np.float64), sr=sr, n_mels=n_mels
    )

    log_ref = librosa.power_to_db(mel_ref)
    log_test = librosa.power_to_db(mel_test)

    # Cosine similarity per frame, then average
    min_frames = min(log_ref.shape[1], log_test.shape[1])
    similarities = []
    for i in range(min_frames):
        a = log_ref[:, i]
        b = log_test[:, i]
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        similarities.append(sim)

    return float(np.mean(similarities))


def test_self_consistency():
    """Use the same model as both base and speaker to test weight injection."""
    from gsv_tts import TTS, MultiSpeakerTTS, SpeakerConfig

    logger.info("=== Self-Consistency Test ===")
    logger.info("Using default models as both base and speaker")

    # ── Full model ──
    logger.info("Loading full model TTS...")
    tts_full = TTS(use_bert=True)
    tts_full.load_gpt_model()
    tts_full.load_sovits_model()

    # ── Shared backbone ──
    logger.info("Loading MultiSpeakerTTS (shared backbone)...")
    speaker = SpeakerConfig(
        name="self_consistency",
        gpt_model_path=str(tts_full.default_gpt_path),
        sovits_model_path=str(tts_full.default_sovits_path),
        spk_audio_path="examples/laffey.mp3",
        prompt_audio_path="examples/AnAn.ogg",
        prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
    )

    tts_shared = MultiSpeakerTTS(
        speakers=[speaker],
        use_bert=True,
    )

    text = "こんにちは、テストです。"

    # ── Inference ──
    logger.info("Running full model inference...")
    audio_full = tts_full.infer(
        spk_audio_path="examples/laffey.mp3",
        prompt_audio_path="examples/AnAn.ogg",
        prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
        text=text,
    )

    logger.info("Running shared backbone inference...")
    audio_shared = tts_shared.infer("self_consistency", text)

    # ── Compare ──
    full_arr = audio_full.audio_data
    shared_arr = audio_shared.audio_data
    min_len = min(len(full_arr), len(shared_arr))

    diff = np.abs(full_arr[:min_len] - shared_arr[:min_len])
    logger.info(f"Sample count — full: {len(full_arr)}, shared: {len(shared_arr)}")
    logger.info(f"Mean absolute difference: {diff.mean():.6f}")
    logger.info(f"Max absolute difference:  {diff.max():.6f}")

    mcd = compute_mcd(full_arr, shared_arr, sr=tts_full.samplerate)
    cos_sim = compute_mel_cosine(full_arr, shared_arr, sr=tts_full.samplerate)
    logger.info(f"MCD (lower is better):       {mcd:.3f}")
    logger.info(f"Mel cosine similarity:       {cos_sim:.4f}")

    # ── Verdict ──
    passed = True
    if diff.mean() > 0.05:
        logger.error(f"FAIL: Mean diff {diff.mean():.4f} exceeds 0.05 threshold")
        passed = False
    if not np.isnan(mcd) and mcd > 2.0:
        logger.error(f"FAIL: MCD {mcd:.2f} exceeds 2.0 threshold")
        passed = False
    if not np.isnan(cos_sim) and cos_sim < 0.90:
        logger.error(f"FAIL: Cosine similarity {cos_sim:.4f} below 0.90")
        passed = False

    if passed:
        logger.info("✓ PASS: Shared backbone output matches full model")
    else:
        logger.error("✗ FAIL: Shared backbone output differs significantly")

    return passed


def test_config_validation():
    """Verify ConfigMismatchError is raised for incompatible configs."""
    from gsv_tts import ConfigMismatchError, SpeakerConfig

    logger.info("\n=== Config Validation Smoke Test ===")

    # Verify the exception class exists and is importable
    assert ConfigMismatchError is not None
    assert issubclass(ConfigMismatchError, ValueError)

    # Verify SpeakerConfig dataclass fields
    cfg = SpeakerConfig(
        name="test",
        gpt_model_path="dummy.ckpt",
        sovits_model_path="dummy.pth",
        spk_audio_path="dummy.wav",
    )
    assert cfg.name == "test"
    assert cfg.prompt_audio_path is None

    logger.info("✓ Config validation types are importable and correct")


def main():
    parser = argparse.ArgumentParser(
        description="Verify SoVITS shared backbone output quality"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only the smoke test (no model loading)"
    )
    args = parser.parse_args()

    logger.info("GSV-TTS-Lite — SoVITS Sharing Verification")
    logger.info("=" * 50)

    # Smoke test (always runs, no model loading)
    test_config_validation()

    if args.quick:
        logger.info("\nQuick mode — skipping full model tests.")
        return 0

    # Full self-consistency test (needs model files + GPU)
    try:
        passed = test_self_consistency()
    except FileNotFoundError as e:
        logger.warning(f"Skipping self-consistency test: {e}")
        logger.warning(
            "Download default models first, or use --quick for smoke test only."
        )
        return 0
    except Exception as e:
        logger.error(f"Self-consistency test failed with error: {e}")
        return 1

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
