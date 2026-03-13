"""
AutoMixAI – Mixing Engine

Synchronizes two audio tracks by BPM, aligns on beat boundaries,
applies a smooth crossfade, and writes the final mix to disk.
"""

import numpy as np
import librosa

from app.services.audio_loader import load_audio, save_audio
from app.services.beat_detector import detect_beats
from app.services.bpm_estimator import estimate_bpm_from_beats
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def time_stretch_to_bpm(
    y: np.ndarray,
    sr: int,
    original_bpm: float,
    target_bpm: float,
) -> np.ndarray:
    """
    Time-stretch *y* so that its tempo matches *target_bpm*.

    Uses ``librosa.effects.time_stretch`` which preserves pitch.

    Args:
        y:            Audio time-series.
        sr:           Sample rate (informational — not used by stretch).
        original_bpm: Current BPM of *y*.
        target_bpm:   Desired BPM after stretching.

    Returns:
        Time-stretched audio array.
    """
    rate = target_bpm / original_bpm
    if abs(rate - 1.0) < 0.005:
        logger.debug("BPM already matched — skipping time-stretch")
        return y

    logger.info(
        "Time-stretching: %.2f → %.2f BPM (rate=%.4f)",
        original_bpm,
        target_bpm,
        rate,
    )
    return librosa.effects.time_stretch(y, rate=rate)


def align_to_beat(
    y: np.ndarray,
    sr: int,
    beat_times: list[float],
) -> np.ndarray:
    """
    Trim the audio so it starts exactly on the first detected beat.

    Args:
        y:          Audio time-series.
        sr:         Sample rate.
        beat_times: Beat timestamps in seconds.

    Returns:
        Trimmed audio starting from the first beat.
    """
    if not beat_times:
        logger.warning("No beats available for alignment — returning original")
        return y

    start_sample = int(beat_times[0] * sr)
    logger.debug("Aligning audio to first beat at %.3f s (sample %d)", beat_times[0], start_sample)
    return y[start_sample:]


def crossfade(
    y1: np.ndarray,
    y2: np.ndarray,
    sr: int,
    duration: float = 5.0,
) -> np.ndarray:
    """
    Apply a linear crossfade between the end of *y1* and the start of
    *y2*.

    Args:
        y1:       Outgoing track.
        y2:       Incoming track.
        sr:       Sample rate.
        duration: Crossfade length in seconds.

    Returns:
        Combined audio array with the crossfade applied.
    """
    fade_samples = int(duration * sr)

    # Clamp fade length so it doesn't exceed either track
    fade_samples = min(fade_samples, len(y1), len(y2))
    if fade_samples < 1:
        logger.warning("Fade region too short — concatenating without fade")
        return np.concatenate([y1, y2])

    # Build linear fade curves
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)

    # Create the crossfade region
    tail = y1[-fade_samples:] * fade_out
    head = y2[:fade_samples] * fade_in
    mixed_region = tail + head

    # Assemble: body of y1 + crossfade + body of y2
    result = np.concatenate([
        y1[:-fade_samples],
        mixed_region,
        y2[fade_samples:],
    ])

    logger.info(
        "Crossfade applied: %.2f s (%d samples)",
        duration,
        fade_samples,
    )
    return result


def create_mix(
    path_a: str,
    path_b: str,
    output_path: str,
    crossfade_duration: float | None = None,
) -> dict:
    """
    End-to-end DJ mixing pipeline.

    Steps:
        1. Load and analyze both tracks (beats + BPM).
        2. Time-stretch both to a common BPM (average).
        3. Align each track to its first beat.
        4. Apply a smooth crossfade.
        5. Save the result.

    Args:
        path_a:             Path to the first (outgoing) track.
        path_b:             Path to the second (incoming) track.
        output_path:        Destination path for the mix.
        crossfade_duration: Crossfade length (seconds).

    Returns:
        Dictionary with mix metadata: ``bpm_a``, ``bpm_b``,
        ``target_bpm``, ``duration``.
    """
    crossfade_duration = crossfade_duration or settings.default_crossfade_duration

    logger.info("=== Creating mix ===")
    logger.info("Track A: %s", path_a)
    logger.info("Track B: %s", path_b)

    # ── 1. Load ───────────────────────────────────────────────────────
    y_a, sr = load_audio(path_a)
    y_b, _ = load_audio(path_b, sr=sr)

    # ── 2. Detect beats & estimate BPM ────────────────────────────────
    beats_a = detect_beats(y_a, sr)
    beats_b = detect_beats(y_b, sr)

    bpm_a = estimate_bpm_from_beats(beats_a) if len(beats_a) >= 2 else 120.0
    bpm_b = estimate_bpm_from_beats(beats_b) if len(beats_b) >= 2 else 120.0

    target_bpm = round((bpm_a + bpm_b) / 2, 2)
    logger.info(
        "BPMs — A: %.2f, B: %.2f → target: %.2f",
        bpm_a,
        bpm_b,
        target_bpm,
    )

    # ── 3. Time-stretch to target BPM ─────────────────────────────────
    y_a = time_stretch_to_bpm(y_a, sr, bpm_a, target_bpm)
    y_b = time_stretch_to_bpm(y_b, sr, bpm_b, target_bpm)

    # ── 4. Align to first beat ────────────────────────────────────────
    y_a = align_to_beat(y_a, sr, beats_a)
    y_b = align_to_beat(y_b, sr, beats_b)

    # ── 5. Crossfade & save ──────────────────────────────────────────
    mixed = crossfade(y_a, y_b, sr, duration=crossfade_duration)
    save_audio(output_path, mixed, sr)

    duration = float(len(mixed) / sr)
    logger.info("Mix complete: %.2f s saved to %s", duration, output_path)

    return {
        "bpm_a": bpm_a,
        "bpm_b": bpm_b,
        "target_bpm": target_bpm,
        "duration": duration,
    }
