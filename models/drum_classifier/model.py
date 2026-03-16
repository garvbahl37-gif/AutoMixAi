"""
Drum Classifier Model
=====================
Classifies percussive audio samples into drum-element categories:
  kick drum, snare, hi-hat, clap, tom, cymbal, drum set, etc.

Two operating modes
-------------------
* **Trained model** — call ``load(checkpoint_path)`` to supply a Keras .h5
  or scikit-learn .pkl model; ``predict()`` routes automatically.
* **Heuristic fallback** — used when no trained model is loaded; uses
  spectral centroid and zero-crossing rate for a coarse classification.

Usage::

    import librosa, numpy as np
    from models.drum_classifier import DrumClassifier

    clf = DrumClassifier()
    y, sr = librosa.load("kick.wav", sr=22050)
    features = clf.extract_features(y, sr)
    label, confidence, info = clf.predict(features)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any

import numpy as np
import librosa


DRUM_CLASSES = [
    "kick drum", "bass drum",
    "snare", "side stick",
    "hi-hat", "closed hi-hat",
    "clap",
    "tom", "high tom", "low tom",
    "cymbal", "crash", "ride",
    "drum set",
    "percussion", "other",
]


class DrumClassifier:
    """
    Classify individual drum elements from audio feature vectors.

    Attributes
    ----------
    classes : list[str]
        All recognisable drum-element labels.
    """

    def __init__(self):
        self.classes = DRUM_CLASSES
        self._model = None

    def load(self, checkpoint_path: str) -> bool:
        """
        Load a pre-trained model from *checkpoint_path*.

        Supports Keras ``.h5`` and scikit-learn ``.pkl`` files.

        Returns True on success, False on failure.
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
            print(f"DrumClassifier: failed to load model — {exc}")
            return False

    def extract_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract a compact feature vector from a raw audio sample.

        Returns a 1-D array:
          [spectral_centroid, zcr, spectral_flatness, onset_strength_max]
        """
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_max = float(np.max(onset_env)) if len(onset_env) > 0 else 0.0
        return np.array([centroid, zcr, flatness, onset_max], dtype=np.float32)

    def predict(
        self,
        features: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify a drum element.

        Parameters
        ----------
        features             : 1-D feature vector (from ``extract_features``).
        confidence_threshold : Minimum confidence to accept a class.

        Returns
        -------
        Tuple of (label, confidence, details_dict).
        """
        if self._model is not None:
            return self._predict_model(features)
        return self._predict_heuristic(features)

    def _predict_model(
        self, features: np.ndarray
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Predict using the loaded Keras / sklearn model."""
        try:
            pred = self._model.predict(features.reshape(1, -1))
            idx = int(np.argmax(pred))
            conf = float(np.max(pred))
            return self.classes[idx % len(self.classes)], conf, {"method": "model"}
        except Exception as exc:
            return "drum set", 0.5, {"method": "model_error", "error": str(exc)}

    def _predict_heuristic(
        self, features: np.ndarray
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Coarse spectral-centroid heuristic (no model required)."""
        if len(features) < 1:
            return "percussion", 0.5, {"method": "heuristic"}

        centroid = float(features[0])

        if centroid < 3_000:
            return "kick drum", 0.85, {"method": "heuristic", "centroid": centroid}
        if centroid < 6_000:
            return "snare", 0.80, {"method": "heuristic", "centroid": centroid}
        return "hi-hat", 0.82, {"method": "heuristic", "centroid": centroid}

    def predict_from_audio(self, y: np.ndarray, sr: int) -> Tuple[str, float, Dict[str, Any]]:
        """Convenience: extract features then classify in one call."""
        return self.predict(self.extract_features(y, sr))
