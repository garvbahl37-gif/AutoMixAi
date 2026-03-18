"""
AutoMixAI – Instrument Classifier Service

Classifies instrument families from audio using the NSynth-trained model.

Families: bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, synth_lead, vocal
"""

from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy-loaded model and scaler
_model = None
_scaler = None
_labels = None
_family_mapping = None

NSYNTH_FAMILIES = ['bass', 'brass', 'flute', 'guitar', 'keyboard',
                   'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']


def _load_model():
    """Load the NSynth instrument classifier model."""
    global _model, _scaler, _labels, _family_mapping

    if _model is not None:
        return True

    model_path = settings.model_dir / "nsynth_classifier.h5"
    scaler_path = settings.model_dir / "nsynth_scaler.pkl"
    labels_path = settings.model_dir / "nsynth_labels.pkl"
    mapping_path = settings.model_dir / "nsynth_family_mapping.pkl"

    if not model_path.exists():
        logger.warning("NSynth classifier model not found at %s", model_path)
        return False

    try:
        from tensorflow import keras
        _model = keras.models.load_model(str(model_path), compile=False)
        logger.info("Loaded NSynth classifier from %s", model_path)

        if scaler_path.exists():
            _scaler = joblib.load(scaler_path)
            logger.info("Loaded NSynth scaler")

        if labels_path.exists():
            _labels = joblib.load(labels_path)
            logger.info("Loaded NSynth labels")

        if mapping_path.exists():
            _family_mapping = joblib.load(mapping_path)
            logger.info("Loaded NSynth family mapping")

        return True
    except Exception as e:
        logger.error("Failed to load NSynth classifier: %s", e)
        return False


def _extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract features compatible with NSynth classifier (55 features)."""
    import librosa

    features = []

    # Spectral features
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spec_flat = librosa.feature.spectral_flatness(y=y)

    features.extend([spec_cent.mean(), spec_cent.std()])
    features.extend([spec_bw.mean(), spec_bw.std()])
    features.extend([spec_rolloff.mean(), spec_rolloff.std()])
    features.extend([spec_flat.mean(), spec_flat.std()])

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend([chroma.mean(), chroma.std()])

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)
    features.extend([zcr.mean(), zcr.std()])

    # RMS
    rms = librosa.feature.rms(y=y)
    features.extend([rms.mean(), rms.std()])

    # MFCCs (20 coefficients, mean and std)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features.extend([mfcc[i].mean(), mfcc[i].std()])

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(tempo))

    return np.array(features, dtype=np.float32)


def classify_instruments(y: np.ndarray, sr: int, top_k: int = 3) -> Dict[str, Any]:
    """
    Classify the dominant instrument family in the audio.

    Args:
        y: Audio signal
        sr: Sample rate
        top_k: Number of top predictions to return

    Returns:
        {
            "instrument": "guitar",
            "confidence": 0.85,
            "top_k": [{"instrument": "guitar", "confidence": 0.85}, ...]
        }
    """
    if not _load_model():
        return {
            "instrument": "unknown",
            "confidence": 0.0,
            "top_k": [],
            "method": "unavailable"
        }

    try:
        # Extract features
        features = _extract_features(y, sr)

        # Scale features
        if _scaler is not None:
            features = _scaler.transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)

        # Predict
        probs = _model.predict(features, verbose=0)[0]

        # Get top-k predictions
        top_indices = np.argsort(probs)[::-1][:top_k]

        # Map indices to family names
        if _family_mapping:
            top_k_results = [
                {"instrument": _family_mapping.get(i, NSYNTH_FAMILIES[i] if i < len(NSYNTH_FAMILIES) else f"class_{i}"),
                 "confidence": round(float(probs[i]), 4)}
                for i in top_indices
            ]
        elif _labels is not None:
            top_k_results = [
                {"instrument": NSYNTH_FAMILIES[_labels.classes_[i]] if i < len(_labels.classes_) else f"class_{i}",
                 "confidence": round(float(probs[i]), 4)}
                for i in top_indices
            ]
        else:
            top_k_results = [
                {"instrument": NSYNTH_FAMILIES[i] if i < len(NSYNTH_FAMILIES) else f"class_{i}",
                 "confidence": round(float(probs[i]), 4)}
                for i in top_indices
            ]

        return {
            "instrument": top_k_results[0]["instrument"],
            "confidence": top_k_results[0]["confidence"],
            "top_k": top_k_results,
            "method": "nsynth_model"
        }

    except Exception as e:
        logger.error("Instrument classification failed: %s", e)
        return {
            "instrument": "unknown",
            "confidence": 0.0,
            "top_k": [],
            "method": "error"
        }
