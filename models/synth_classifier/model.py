"""
Synth Classifier Model
=======================
Classifies synthesizer and electronic sound sources into:
  synth lead, synth bass, synth pad, synth keys, string synth,
  choir pad, electronic piano, digital instrument, sampler, soundscape.

Also estimates timbre characteristics: bright / dark / hollow / full.

Usage::

    from models.synth_classifier import SynthClassifier
    import librosa

    clf = SynthClassifier()
    y, sr = librosa.load("synth_lead.wav", sr=22050)
    result = clf.predict_from_audio(y, sr)
    print(result["synth_class"], result["timbre"])
    # → "synth lead"  "bright"
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import librosa


SYNTH_CLASSES = [
    "synth lead",
    "synth bass",
    "synth pad",
    "synth keys",
    "string synth",
    "choir pad",
    "electronic piano",
    "digital instrument",
    "sampler",
    "soundscape",
]

TIMBRE_CLASSES = ["bright", "dark", "hollow", "full", "brittle", "smooth"]


class SynthClassifier:
    """
    Identify synthesiser type and timbral characteristic from audio.

    Parameters
    ----------
    n_mfcc : int
        Number of MFCCs used for feature extraction (default 13).
    """

    def __init__(self, n_mfcc: int = 13):
        self.n_mfcc = n_mfcc
        self.classes = SYNTH_CLASSES
        self.timbre_classes = TIMBRE_CLASSES
        self._model = None

    def load(self, checkpoint_path: str) -> bool:
        """Load a trained Keras (.h5) or scikit-learn (.pkl) model."""
        try:
            if checkpoint_path.endswith(".h5"):
                from tensorflow import keras
                self._model = keras.models.load_model(checkpoint_path)
            else:
                import joblib
                self._model = joblib.load(checkpoint_path)
            return True
        except Exception as exc:
            print(f"SynthClassifier: load failed — {exc}")
            return False

    def extract_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract synth-focused features.

        Feature layout:
          spectral_centroid, spectral_spread, spectral_flatness,
          mfcc_mean(13), zcr, rms
          = 18 dims total.
        """
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))

        # Spectral spread (weighted std dev of frequencies around centroid)
        power = S ** 2
        power_sum = power.sum(axis=0, keepdims=True) + 1e-8
        spread = float(np.mean(
            np.sqrt(((freqs[:, None] - centroid) ** 2 * power / power_sum).sum(axis=0))
        ))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc).mean(axis=1)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        rms = float(np.mean(librosa.feature.rms(y=y)))

        return np.concatenate([[centroid, spread, flatness], mfcc, [zcr, rms]]).astype(np.float32)

    def predict(self, features: np.ndarray, return_timbre: bool = True) -> Dict[str, Any]:
        """
        Classify a synth sound from a feature vector.

        Parameters
        ----------
        features      : 1-D feature vector from ``extract_features``.
        return_timbre : Include timbre classification in result.

        Returns
        -------
        Dict with keys:
          is_synth, synth_class, confidence, timbre, timbre_confidence.
        """
        if self._model is not None:
            return self._predict_model(features, return_timbre)
        return self._predict_heuristic(features, return_timbre)

    def predict_from_audio(self, y: np.ndarray, sr: int, return_timbre: bool = True) -> Dict[str, Any]:
        """Convenience: extract features then classify."""
        return self.predict(self.extract_features(y, sr), return_timbre)

    def _predict_model(self, features: np.ndarray, return_timbre: bool) -> Dict[str, Any]:
        try:
            probs = self._model.predict(features.reshape(1, -1))[0]
            idx = int(np.argmax(probs))
            return {
                "is_synth": True,
                "synth_class": self.classes[idx % len(self.classes)],
                "confidence": float(probs[idx]),
                "timbre": self._timbre_from_features(features) if return_timbre else None,
                "timbre_confidence": 0.7,
            }
        except Exception:
            return self._predict_heuristic(features, return_timbre)

    def _predict_heuristic(self, features: np.ndarray, return_timbre: bool) -> Dict[str, Any]:
        """Heuristic: spectral centroid + spread → synth type."""
        centroid = float(features[0]) if len(features) > 0 else 2_000.0
        spread   = float(features[1]) if len(features) > 1 else 1_000.0

        if centroid < 500:
            synth_class, conf = "synth bass", 0.80
        elif spread > 4_000:
            synth_class, conf = "synth pad", 0.75
        elif centroid > 5_000:
            synth_class, conf = "synth lead", 0.78
        else:
            synth_class, conf = "synth keys", 0.70

        timbre = self._timbre_from_features(features) if return_timbre else None
        return {
            "is_synth": True,
            "synth_class": synth_class,
            "confidence": conf,
            "timbre": timbre,
            "timbre_confidence": 0.65,
        }

    def _timbre_from_features(self, features: np.ndarray) -> str:
        """Estimate timbre label from spectral centroid (features[0])."""
        centroid = float(features[0]) if len(features) > 0 else 2_000.0
        if centroid > 6_000:
            return "bright"
        if centroid > 3_000:
            return "full"
        if centroid > 1_500:
            return "smooth"
        return "dark"
