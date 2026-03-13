"""
AutoMixAI – Model Inference

Loads a trained Keras model and converts frame-level beat-probability
predictions into beat timestamps.
"""

from pathlib import Path
from functools import lru_cache

import numpy as np
from tensorflow import keras
import librosa

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_model(path: str | Path | None = None) -> keras.Model:
    """
    Load the trained ANN model from disk (cached after first call).

    Args:
        path: Path to the ``.h5`` model file.  Defaults to
              ``settings.model_path``.

    Returns:
        Compiled Keras Model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = Path(path or settings.model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run `python -m app.model.train` first."
        )

    logger.info("Loading trained model from %s", model_path)
    model = keras.models.load_model(str(model_path))
    return model


def predict_beats(model: keras.Model, features: np.ndarray) -> np.ndarray:
    """
    Run the ANN forward pass to get per-frame beat probabilities.

    Args:
        model:    Loaded Keras model.
        features: Feature matrix of shape ``(n_frames, n_features)``.

    Returns:
        1-D array of probabilities in ``[0, 1]``.
    """
    predictions = model.predict(features, verbose=0).flatten()
    logger.debug(
        "Predictions — min=%.4f, max=%.4f, mean=%.4f",
        predictions.min(),
        predictions.max(),
        predictions.mean(),
    )
    return predictions


def predictions_to_timestamps(
    predictions: np.ndarray,
    sr: int | None = None,
    hop_length: int | None = None,
    threshold: float | None = None,
) -> list[float]:
    """
    Convert beat-probability predictions to beat timestamps.

    Frames whose probability exceeds *threshold* are considered beats.
    Consecutive beat frames are merged (only the first frame of each
    onset is retained).

    Args:
        predictions: 1-D probability array from :func:`predict_beats`.
        sr:          Sample rate.
        hop_length:  Hop length used during feature extraction.
        threshold:   Probability cut-off for a positive classification.

    Returns:
        Sorted list of beat times in seconds.
    """
    sr = sr or settings.sample_rate
    hop_length = hop_length or settings.hop_length
    threshold = threshold if threshold is not None else settings.beat_threshold

    # Threshold the predictions
    beat_frames = np.where(predictions >= threshold)[0]

    if len(beat_frames) == 0:
        logger.warning("No beats detected at threshold %.2f", threshold)
        return []

    # Merge consecutive frames — keep only the onset of each beat region
    merged: list[int] = [beat_frames[0]]
    for frame in beat_frames[1:]:
        if frame - merged[-1] > 1:
            merged.append(frame)

    beat_times = librosa.frames_to_time(
        merged, sr=sr, hop_length=hop_length
    ).tolist()

    logger.info("Detected %d beats (threshold=%.2f)", len(beat_times), threshold)
    return beat_times
