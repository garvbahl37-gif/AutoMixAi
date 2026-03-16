"""
Beat Detection Model
====================
Wraps the trained ANN for per-frame beat classification.

Expected model input  : (n_frames, 43)  — 43-dim feature vector
Expected model output : (n_frames, 1)   — beat probability in [0, 1]

Usage::

    model = BeatDetectionModel("path/to/beat_detector.h5")
    beat_times = model.predict(audio_signal, sample_rate)
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import numpy as np
import librosa


# Hop length used during inference (must match training)
INFERENCE_HOP = 1024
BEAT_THRESHOLD = 0.5


def _extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract 43-dim feature matrix from audio signal."""
    n_fft = 2048
    n_mfcc = 13

    S_complex = librosa.stft(y, n_fft=n_fft, hop_length=INFERENCE_HOP)
    S_mag = np.abs(S_complex)
    T = S_mag.shape[1]

    mel = librosa.feature.melspectrogram(S=S_mag ** 2, sr=sr, n_fft=n_fft)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(S=S_mag ** 2, sr=sr, hop_length=INFERENCE_HOP, n_fft=n_fft)
    onset = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S_mag, ref=np.max), sr=sr, hop_length=INFERENCE_HOP)
    flux = np.concatenate([[0.0], np.sqrt(np.sum(np.diff(S_mag, axis=1) ** 2, axis=0))])
    rms = librosa.feature.rms(S=S_mag, frame_length=n_fft, hop_length=INFERENCE_HOP)[0]
    centroid = librosa.feature.spectral_centroid(S=S_mag, sr=sr, hop_length=INFERENCE_HOP, n_fft=n_fft)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=INFERENCE_HOP)[0]

    n_frames = min(T, mfcc.shape[1], delta_mfcc.shape[1], chroma.shape[1],
                   len(onset), len(flux), len(rms), len(centroid), len(zcr))

    return np.column_stack([
        mfcc.T[:n_frames],
        delta_mfcc.T[:n_frames],
        chroma.T[:n_frames],
        onset[:n_frames],
        flux[:n_frames],
        rms[:n_frames],
        centroid[:n_frames],
        zcr[:n_frames],
    ])


class BeatDetectionModel:
    """
    ANN-based beat detector.

    Parameters
    ----------
    model_path : str or Path
        Path to the trained ``beat_detector.h5`` Keras model file.
    threshold : float
        Probability threshold for a frame to be classified as a beat.
    """

    def __init__(self, model_path: str | Path, threshold: float = BEAT_THRESHOLD):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self._model = None

    def load(self) -> None:
        """Load the Keras model from disk (lazy — called automatically on first predict)."""
        if self._model is not None:
            return
        try:
            from tensorflow import keras
            self._model = keras.models.load_model(str(self.model_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to load beat detection model: {exc}") from exc

    def predict(self, y: np.ndarray, sr: int) -> list[float]:
        """
        Run beat detection on a mono audio signal.

        Parameters
        ----------
        y  : Audio time-series (mono, normalised).
        sr : Sample rate.

        Returns
        -------
        Sorted list of beat timestamps in seconds.
        Falls back to librosa beat tracker if the model file is missing.
        """
        if not self.model_path.exists():
            return self._librosa_fallback(y, sr)

        self.load()
        features = _extract_features(y, sr)

        expected_dim = self._model.input_shape[-1]
        if features.shape[1] != expected_dim:
            raise ValueError(
                f"Feature dim mismatch: model expects {expected_dim}, got {features.shape[1]}"
            )

        probs = self._model.predict(features, verbose=0).flatten()
        beat_frames_raw = np.where(probs >= self.threshold)[0]

        if len(beat_frames_raw) == 0:
            return []

        # Merge consecutive frames (keep only onset of each beat region)
        merged: list[int] = [beat_frames_raw[0]]
        for f in beat_frames_raw[1:]:
            if f - merged[-1] > 1:
                merged.append(f)

        return librosa.frames_to_time(merged, sr=sr, hop_length=INFERENCE_HOP).tolist()

    @staticmethod
    def _librosa_fallback(y: np.ndarray, sr: int) -> list[float]:
        """Fallback beat detection using librosa when no model is available."""
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        return librosa.frames_to_time(beat_frames, sr=sr, hop_length=512).tolist()
