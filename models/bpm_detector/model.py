"""
BPM Detector Model
==================
Estimates tempo (BPM) from either:
  1. A list of beat timestamps (preferred — more accurate)
  2. Raw audio (librosa-based fallback)

Usage::

    detector = BPMDetector()

    # From beat timestamps
    bpm = detector.from_beats([0.5, 1.0, 1.5, 2.0])   # → 120.0

    # From raw audio
    bpm = detector.from_audio(y, sr)
"""

from __future__ import annotations

import numpy as np
import librosa


class BPMDetector:
    """
    Tempo estimator supporting both beat-timestamp and raw-audio inputs.

    Parameters
    ----------
    hop_length : int
        Hop length used by the librosa fallback (default 512).
    """

    def __init__(self, hop_length: int = 512):
        self.hop_length = hop_length

    def from_beats(self, beat_times: list[float]) -> float:
        """
        Estimate BPM from a sorted list of beat timestamps (seconds).

        Uses the *median* inter-beat interval — robust to outlier beats.

        Parameters
        ----------
        beat_times : Sorted list of beat timestamps.

        Returns
        -------
        float : BPM rounded to 2 decimal places.

        Raises
        ------
        ValueError : If fewer than 2 beats are supplied, or IBIs are invalid.
        """
        if len(beat_times) < 2:
            raise ValueError("At least 2 beat timestamps are required to estimate BPM.")

        ibis = np.diff(beat_times)
        median_ibi = float(np.median(ibis))

        if median_ibi <= 0:
            raise ValueError("Invalid inter-beat intervals — median IBI must be > 0.")

        return round(60.0 / median_ibi, 2)

    def from_audio(self, y: np.ndarray, sr: int) -> float:
        """
        Estimate BPM directly from an audio signal using librosa's beat tracker.

        Parameters
        ----------
        y  : Mono audio time-series.
        sr : Sample rate.

        Returns
        -------
        float : Estimated BPM.
        """
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        return round(float(np.asarray(tempo).flat[0]), 2)

    def estimate(self, y: np.ndarray, sr: int, beat_times: list[float] | None = None) -> float:
        """
        Unified entry point: uses beat timestamps if provided, else raw audio.

        Parameters
        ----------
        y          : Mono audio time-series.
        sr         : Sample rate.
        beat_times : Optional pre-computed beat timestamps.

        Returns
        -------
        float : Estimated BPM.
        """
        if beat_times and len(beat_times) >= 2:
            return self.from_beats(beat_times)
        return self.from_audio(y, sr)
