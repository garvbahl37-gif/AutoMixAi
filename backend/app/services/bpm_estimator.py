"""
AutoMixAI – BPM Estimator Service

Estimates tempo from detected beat timestamps or directly from audio
via librosa.
"""

import numpy as np
import librosa

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def estimate_bpm_from_beats(beat_times: list[float]) -> float:
    """
    Estimate BPM from a list of beat timestamps.

    The tempo is derived from the **median** of inter-beat intervals
    (IBIs), which is more robust to outliers than the mean.

    Args:
        beat_times: Sorted beat timestamps in seconds.

    Returns:
        Estimated BPM as a float.

    Raises:
        ValueError: If fewer than 2 beats are provided.
    """
    if len(beat_times) < 2:
        raise ValueError(
            "At least 2 beat timestamps are required to estimate BPM."
        )

    ibis = np.diff(beat_times)                 # inter-beat intervals
    median_ibi = float(np.median(ibis))

    if median_ibi <= 0:
        raise ValueError("Invalid inter-beat intervals (median ≤ 0).")

    bpm = 60.0 / median_ibi

    logger.info(
        "BPM estimated from %d beats: %.2f BPM (median IBI=%.4f s)",
        len(beat_times),
        bpm,
        median_ibi,
    )
    return round(bpm, 2)


def estimate_bpm_librosa(y: np.ndarray, sr: int) -> float:
    """
    Fallback BPM estimation using librosa's built-in beat tracker.

    Args:
        y:  Audio time-series (mono).
        sr: Sample rate.

    Returns:
        Estimated BPM.
    """
    tempo, _ = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=settings.hop_length
    )
    bpm = float(np.asarray(tempo).flat[0])

    logger.info("librosa BPM estimation: %.2f BPM", bpm)
    return round(bpm, 2)
