"""
AutoMixAI – Music Tag Predictor Service

Predicts multi-label music tags using the MagnaTagATune-trained model.

Tags include: guitar, piano, drums, vocal, fast, slow, loud, quiet,
              rock, electronic, classical, jazz, etc.
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
_tag_labels = None


def _load_model():
    """Load the tag predictor model."""
    global _model, _scaler, _tag_labels

    if _model is not None:
        return True

    model_path = settings.model_dir / "tag_predictor.h5"
    scaler_path = settings.model_dir / "tag_scaler.pkl"
    labels_path = settings.model_dir / "tag_labels.pkl"

    if not model_path.exists():
        logger.warning("Tag predictor model not found at %s", model_path)
        return False

    try:
        from tensorflow import keras
        _model = keras.models.load_model(str(model_path), compile=False)
        logger.info("Loaded tag predictor from %s", model_path)

        if scaler_path.exists():
            _scaler = joblib.load(scaler_path)
            logger.info("Loaded tag scaler")

        if labels_path.exists():
            _tag_labels = joblib.load(labels_path)
            logger.info("Loaded %d tag labels", len(_tag_labels))

        return True
    except Exception as e:
        logger.error("Failed to load tag predictor: %s", e)
        return False


def _extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract features compatible with tag predictor (55 features)."""
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


def predict_tags(y: np.ndarray, sr: int, threshold: float = 0.3, max_tags: int = 10) -> Dict[str, Any]:
    """
    Predict music tags for the audio.

    Args:
        y: Audio signal
        sr: Sample rate
        threshold: Minimum confidence to include a tag
        max_tags: Maximum number of tags to return

    Returns:
        {
            "tags": ["guitar", "rock", "male vocal", ...],
            "tag_scores": [{"tag": "guitar", "score": 0.85}, ...],
            "mood": "energetic",
            "has_vocals": True,
            "is_instrumental": False
        }
    """
    if not _load_model():
        return _heuristic_tags(y, sr)

    try:
        # Extract features
        features = _extract_features(y, sr)

        # Scale features
        if _scaler is not None:
            features = _scaler.transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)

        # Predict (sigmoid outputs)
        probs = _model.predict(features, verbose=0)[0]

        # Get tags above threshold
        if _tag_labels is not None:
            tag_scores = [
                {"tag": _tag_labels[i], "score": round(float(probs[i]), 4)}
                for i in range(len(probs))
                if probs[i] >= threshold
            ]
        else:
            tag_scores = [
                {"tag": f"tag_{i}", "score": round(float(probs[i]), 4)}
                for i in range(len(probs))
                if probs[i] >= threshold
            ]

        # Sort by score
        tag_scores = sorted(tag_scores, key=lambda x: x["score"], reverse=True)[:max_tags]
        tags = [t["tag"] for t in tag_scores]

        # Derive mood and vocal detection from tags
        mood = _derive_mood(tags)
        has_vocals = _has_vocals(tags)

        return {
            "tags": tags,
            "tag_scores": tag_scores,
            "mood": mood,
            "has_vocals": has_vocals,
            "is_instrumental": not has_vocals,
            "method": "tag_model"
        }

    except Exception as e:
        logger.error("Tag prediction failed: %s", e)
        return _heuristic_tags(y, sr)


def _derive_mood(tags: List[str]) -> str:
    """Derive mood from predicted tags."""
    tags_lower = [t.lower() for t in tags]

    if any(t in tags_lower for t in ["happy", "upbeat", "fast", "energetic"]):
        return "energetic"
    elif any(t in tags_lower for t in ["sad", "slow", "melancholic"]):
        return "melancholic"
    elif any(t in tags_lower for t in ["calm", "quiet", "ambient"]):
        return "calm"
    elif any(t in tags_lower for t in ["loud", "heavy", "hard"]):
        return "intense"
    else:
        return "neutral"


def _has_vocals(tags: List[str]) -> bool:
    """Check if tags indicate presence of vocals."""
    tags_lower = [t.lower() for t in tags]
    vocal_tags = ["vocal", "vocals", "voice", "singing", "singer", "male vocal",
                  "female vocal", "choir", "rap", "male voice", "female voice"]
    no_vocal_tags = ["no vocal", "no vocals", "no voice", "instrumental"]

    if any(t in tags_lower for t in no_vocal_tags):
        return False
    return any(t in tags_lower for t in vocal_tags)


def _heuristic_tags(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Fallback heuristic tag prediction."""
    import librosa

    tags = []
    tag_scores = []

    # RMS for energy
    rms = librosa.feature.rms(y=y).mean()
    if rms > 0.1:
        tags.append("loud")
        tag_scores.append({"tag": "loud", "score": min(1.0, rms * 5)})
    else:
        tags.append("quiet")
        tag_scores.append({"tag": "quiet", "score": max(0.3, 1 - rms * 5)})

    # Tempo for speed
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if tempo > 130:
        tags.append("fast")
        tag_scores.append({"tag": "fast", "score": 0.7})
    elif tempo < 90:
        tags.append("slow")
        tag_scores.append({"tag": "slow", "score": 0.7})

    # Spectral centroid for brightness
    cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    if cent > 3000:
        tags.append("bright")
    else:
        tags.append("dark")

    return {
        "tags": tags,
        "tag_scores": tag_scores,
        "mood": "energetic" if tempo > 120 else "calm",
        "has_vocals": False,  # Can't detect without model
        "is_instrumental": True,
        "method": "heuristic"
    }
