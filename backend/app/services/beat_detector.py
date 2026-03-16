"""
AutoMixAI – Beat Detector Service

Provides beat detection via the trained ANN model with a librosa-based
fallback when no model is available.
"""

import numpy as np

from app.services.audio_loader import load_audio
from app.services.feature_extractor import extract_features
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def detect_beats(
    y: np.ndarray,
    sr: int,
    model_path: str | None = None,
) -> list[float]:
    """
    Detect beat timestamps using the trained ANN model.

    The audio signal is converted into a feature matrix, passed through
    the model, and the resulting probabilities are thresholded to
    produce timestamps.

    Falls back to :func:`detect_beats_librosa` when the model file is
    not found or TensorFlow is not installed.

    Args:
        y:          Audio time-series (mono, normalized).
        sr:         Sample rate.
        model_path: Optional override for the model ``.h5`` path.

    Returns:
        Sorted list of beat timestamps (seconds).
    """
    try:
        from app.model.inference import load_model, predict_beats, predictions_to_timestamps
    except ImportError as exc:
        logger.warning(
            "TensorFlow not available (%s) — falling back to librosa beat tracker", exc
        )
        return detect_beats_librosa(y, sr)

    model_file = model_path or str(settings.model_path)

    try:
        model = load_model(model_file)
    except FileNotFoundError:
        logger.warning(
            "No trained model found — falling back to librosa beat tracker"
        )
        return detect_beats_librosa(y, sr)

    features = extract_features(y, sr)
    try:
        predictions = predict_beats(model, features)
    except ValueError as exc:
        logger.warning(
            "Model/feature mismatch (%s) — falling back to librosa beat tracker", exc
        )
        return detect_beats_librosa(y, sr)
    beat_times = predictions_to_timestamps(predictions, sr=sr)

    logger.info("ANN beat detection returned %d beats", len(beat_times))
    return beat_times


def detect_beats_librosa(y: np.ndarray, sr: int) -> list[float]:
    """
    Fallback beat detection using the librosa beat tracker.

    This function does **not** require a trained model and is used
    when the ANN model has not been trained yet.

    Args:
        y:  Audio time-series.
        sr: Sample rate.

    Returns:
        Sorted list of beat timestamps (seconds).
    """
    import librosa

    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=settings.hop_length
    )
    beat_times = librosa.frames_to_time(
        beat_frames, sr=sr, hop_length=settings.hop_length
    ).tolist()

    logger.info(
        "librosa beat detection: %d beats, estimated tempo=%.1f BPM",
        len(beat_times),
        float(np.asarray(tempo).flat[0]),
    )
    return beat_times
