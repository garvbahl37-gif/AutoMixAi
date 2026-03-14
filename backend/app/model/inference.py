"""
AutoMixAI – Model Inference

Loads a trained Keras model and converts frame-level beat-probability
predictions into beat timestamps.

v2 changes:
  - Auto-loads ``feature_scaler.pkl`` (sklearn StandardScaler) if present
    beside the model file.  Required when the v2 notebook was used for
    training; silently skipped for the v1 model.
  - Model input dimensionality is inferred from the loaded model so no
    manual config change is needed when swapping v1 ↔ v2.
"""

from pathlib import Path
from functools import lru_cache

import numpy as np
from tensorflow import keras
import librosa

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Model loading ─────────────────────────────────────────────────────────────

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
            "Run the Kaggle notebook and place beat_detector.h5 in "
            "backend/app/storage/models/."
        )

    logger.info("Loading trained model from %s", model_path)
    model = keras.models.load_model(str(model_path))
    input_dim = model.input_shape[-1]
    logger.info("Model loaded — input_dim=%d", input_dim)
    return model


# ── Feature scaler (v2 only) ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_scaler(path: str | Path | None = None):
    """
    Load the sklearn StandardScaler saved by the v2 training notebook.

    Returns ``None`` silently if the scaler file is not found (v1 model
    compatibility).

    Args:
        path: Path to ``feature_scaler.pkl``.  Defaults to
              ``settings.scaler_path``.
    """
    scaler_path = Path(path or settings.scaler_path)

    if not scaler_path.exists():
        logger.info(
            "No feature scaler found at %s — assuming raw (unscaled) features.",
            scaler_path,
        )
        return None

    try:
        import joblib
        scaler = joblib.load(str(scaler_path))
        logger.info("Feature scaler loaded from %s", scaler_path)
        return scaler
    except Exception as exc:
        logger.warning("Could not load scaler (%s) — skipping scaling.", exc)
        return None


# ── Inference helpers ─────────────────────────────────────────────────────────

def predict_beats(model: keras.Model, features: np.ndarray) -> np.ndarray:
    """
    Run the ANN forward pass to get per-frame beat probabilities.

    Automatically applies the v2 StandardScaler when ``feature_scaler.pkl``
    is present in the models directory.

    Args:
        model:    Loaded Keras model.
        features: Feature matrix of shape ``(n_frames, n_features)``.

    Returns:
        1-D array of probabilities in ``[0, 1]``.
    """
    scaler = load_scaler()
    if scaler is not None:
        features = scaler.transform(features)

    # Guard: model expects a specific input dimension
    expected_dim = model.input_shape[-1]
    if features.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: model expects {expected_dim} dims "
            f"but got {features.shape[1]}. Make sure the feature extractor "
            "matches the training notebook (v1=16, v2=43)."
        )

    predictions = model.predict(features, verbose=0).flatten()
    logger.debug(
        "Predictions — min=%.4f  max=%.4f  mean=%.4f",
        predictions.min(), predictions.max(), predictions.mean(),
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
    from app.services.feature_extractor import INFERENCE_HOP

    sr         = sr or settings.sample_rate
    hop_length = hop_length or INFERENCE_HOP   # use inference hop, not training hop
    threshold  = threshold if threshold is not None else settings.beat_threshold

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
