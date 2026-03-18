"""
AutoMixAI – Drum Classifier Service

Classifies drum sounds into categories: kick, snare, hihat, tom, etc.
Useful for analyzing drum patterns and identifying percussion elements.
"""

from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Any

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy-loaded model and scaler
_model = None
_scaler = None
_labels = None


def _load_model():
    """Load the drum classifier model."""
    global _model, _scaler, _labels

    if _model is not None:
        return True

    model_path = settings.model_dir / "drum_classifier.h5"
    scaler_path = settings.model_dir / "drum_scaler.pkl"
    labels_path = settings.model_dir / "drum_labels.pkl"

    if not model_path.exists():
        logger.warning("Drum classifier model not found at %s", model_path)
        return False

    try:
        from tensorflow import keras
        _model = keras.models.load_model(str(model_path), compile=False)
        logger.info("Loaded drum classifier from %s", model_path)

        if scaler_path.exists():
            _scaler = joblib.load(scaler_path)
            logger.info("Loaded drum scaler")

        if labels_path.exists():
            _labels = joblib.load(labels_path)
            logger.info("Loaded drum labels: %s", list(_labels.classes_))

        return True
    except Exception as e:
        logger.error("Failed to load drum classifier: %s", e)
        return False


def _extract_drum_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract features optimized for drum classification (55 features)."""
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

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features.extend([mfcc[i].mean(), mfcc[i].std()])

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(tempo))

    return np.array(features, dtype=np.float32)


def classify_drum(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Classify a drum sound sample.

    Args:
        y: Audio signal (should be a short drum hit)
        sr: Sample rate

    Returns:
        {
            "drum_type": "kick",
            "confidence": 0.95,
            "all_classes": [{"type": "kick", "score": 0.95}, ...]
        }
    """
    if not _load_model():
        return {
            "drum_type": "unknown",
            "confidence": 0.0,
            "all_classes": [],
            "method": "unavailable"
        }

    try:
        features = _extract_drum_features(y, sr)

        if _scaler is not None:
            features = _scaler.transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)

        probs = _model.predict(features, verbose=0)[0]
        top_idx = np.argmax(probs)

        if _labels is not None:
            drum_type = str(_labels.classes_[top_idx])
            all_classes = [
                {"type": str(_labels.classes_[i]), "score": round(float(probs[i]), 4)}
                for i in range(len(probs))
            ]
        else:
            drum_types = ["kick", "snare", "hihat", "tom"]
            drum_type = drum_types[top_idx] if top_idx < len(drum_types) else f"class_{top_idx}"
            all_classes = [
                {"type": drum_types[i] if i < len(drum_types) else f"class_{i}",
                 "score": round(float(probs[i]), 4)}
                for i in range(len(probs))
            ]

        all_classes = sorted(all_classes, key=lambda x: x["score"], reverse=True)

        return {
            "drum_type": drum_type,
            "confidence": round(float(probs[top_idx]), 4),
            "all_classes": all_classes,
            "method": "drum_model"
        }

    except Exception as e:
        logger.error("Drum classification failed: %s", e)
        return {
            "drum_type": "unknown",
            "confidence": 0.0,
            "all_classes": [],
            "method": "error"
        }


def analyze_drum_pattern(y: np.ndarray, sr: int, beat_times: List[float]) -> Dict[str, Any]:
    """
    Analyze the drum pattern in a full track by classifying segments around beats.

    Args:
        y: Full audio signal
        sr: Sample rate
        beat_times: List of beat timestamps

    Returns:
        {
            "dominant_drums": ["kick", "snare"],
            "pattern_analysis": {...},
            "drum_density": 0.7
        }
    """
    if not _load_model() or len(beat_times) < 4:
        return {
            "dominant_drums": [],
            "pattern_analysis": {},
            "drum_density": 0.0
        }

    # Analyze short segments around each beat
    drum_counts = {}
    segment_duration = 0.1  # 100ms around each beat

    for beat_time in beat_times[:32]:  # Analyze first 32 beats
        start_sample = max(0, int((beat_time - 0.02) * sr))
        end_sample = min(len(y), int((beat_time + segment_duration) * sr))

        if end_sample - start_sample < sr * 0.05:
            continue

        segment = y[start_sample:end_sample]
        result = classify_drum(segment, sr)

        if result["confidence"] > 0.5:
            drum_type = result["drum_type"]
            drum_counts[drum_type] = drum_counts.get(drum_type, 0) + 1

    # Sort by frequency
    sorted_drums = sorted(drum_counts.items(), key=lambda x: x[1], reverse=True)
    dominant = [d[0] for d in sorted_drums[:3]]

    # Calculate density (drums per beat)
    total_drums = sum(drum_counts.values())
    density = total_drums / max(1, len(beat_times[:32]))

    return {
        "dominant_drums": dominant,
        "pattern_analysis": drum_counts,
        "drum_density": round(density, 2)
    }
