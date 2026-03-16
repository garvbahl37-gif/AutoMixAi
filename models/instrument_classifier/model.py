"""
Instrument Classifier Model
============================
General-purpose instrument type classifier using MFCC + spectral features.

Classifies audio stems into broad instrument families:
  drums/percussion, bass, guitar, piano/keys, strings, brass/wind,
  vocal, synth, other.

Usage::

    from models.instrument_classifier import InstrumentClassifier
    import librosa

    clf = InstrumentClassifier()
    y, sr = librosa.load("stem.wav", sr=22050)
    label, confidence = clf.predict_from_audio(y, sr)
    # → ("synth", 0.78)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any

import numpy as np
import librosa


INSTRUMENT_FAMILIES = [
    "drums",
    "bass",
    "guitar",
    "piano",
    "strings",
    "brass",
    "vocal",
    "synth",
    "other",
]

# Rough spectral-centroid ranges per family (Hz) — used for heuristic fallback
_CENTROID_RANGES: Dict[str, Tuple[float, float]] = {
    "bass":    (80,   600),
    "drums":   (200,  8_000),
    "guitar":  (500,  3_500),
    "piano":   (300,  5_000),
    "strings": (200,  5_000),
    "brass":   (300,  4_000),
    "vocal":   (200,  4_000),
    "synth":   (100, 12_000),
}


class InstrumentClassifier:
    """
    Classify audio stems into instrument families.

    Parameters
    ----------
    n_mfcc : int
        Number of MFCCs to compute (default 20).
    """

    def __init__(self, n_mfcc: int = 20):
        self.n_mfcc = n_mfcc
        self.classes = INSTRUMENT_FAMILIES
        self._model = None

    def load(self, checkpoint_path: str) -> bool:
        """
        Load a trained Keras (.h5) or scikit-learn (.pkl) model.

        Returns True on success.
        """
        try:
            if checkpoint_path.endswith(".h5"):
                from tensorflow import keras
                self._model = keras.models.load_model(checkpoint_path)
            else:
                import joblib
                self._model = joblib.load(checkpoint_path)
            return True
        except Exception as exc:
            print(f"InstrumentClassifier: load failed — {exc}")
            return False

    def extract_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract a compact feature vector from an audio stem.

        Feature layout:
          [mfcc_mean(20), spectral_centroid, spectral_rolloff, zcr, rms]
          = 24 dims total.
        """
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc).mean(axis=1)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        rms = float(np.mean(librosa.feature.rms(y=y)))

        return np.concatenate([mfcc, [centroid, rolloff, zcr, rms]]).astype(np.float32)

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Classify an instrument from a feature vector.

        Returns
        -------
        Tuple of (family_label, confidence).
        """
        if self._model is not None:
            return self._predict_model(features)
        return self._predict_heuristic(features)

    def predict_from_audio(self, y: np.ndarray, sr: int) -> Tuple[str, float]:
        """Convenience: extract features then classify."""
        return self.predict(self.extract_features(y, sr))

    def predict_all_probs(self, features: np.ndarray) -> Dict[str, float]:
        """
        Return a probability dict for all instrument families.
        Falls back to uniform distribution when no model is loaded.
        """
        if self._model is not None:
            try:
                probs = self._model.predict(features.reshape(1, -1))[0]
                return {cls: float(p) for cls, p in zip(self.classes, probs)}
            except Exception:
                pass
        return {cls: 1.0 / len(self.classes) for cls in self.classes}

    def _predict_model(self, features: np.ndarray) -> Tuple[str, float]:
        try:
            probs = self._model.predict(features.reshape(1, -1))[0]
            idx = int(np.argmax(probs))
            return self.classes[idx % len(self.classes)], float(probs[idx])
        except Exception:
            return "other", 0.5

    def _predict_heuristic(self, features: np.ndarray) -> Tuple[str, float]:
        """Use spectral centroid (features[-4]) as a coarse heuristic."""
        centroid = float(features[-4]) if len(features) >= 4 else 1_000.0

        if centroid < 400:
            return "bass", 0.75
        if centroid < 800:
            return "drums", 0.65
        if centroid < 2_500:
            return "guitar", 0.60
        if centroid < 5_000:
            return "piano", 0.60
        return "synth", 0.55
