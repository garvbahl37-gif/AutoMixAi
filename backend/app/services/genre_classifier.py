"""
AutoMixAI – Genre Classifier Service

Extracts a 57-dimensional feature vector matching the GTZAN dataset CSV
(features_30_sec / features_3_sec) for model compatibility, then classifies
into one of 10 music genres:
  blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

Feature layout (57 dim, matching GTZAN CSV column order):
  chroma_stft mean+var    (2)
  rms mean+var            (2)
  spectral_centroid …var  (2)
  spectral_bandwidth …var (2)
  rolloff …var            (2)
  zero_crossing_rate …var (2)
  harmony …var            (2)
  perceptr …var           (2)
  tempo                   (1)
  mfcc1..mfcc20 mean+var  (40)
  ────────────────────────────
  TOTAL                   57

Two modes:
  1. Model-based  — loads genre_classifier.h5 + genre_scaler.pkl if present
  2. Heuristic    — rule / scoring fallback (no model needed)
"""

from __future__ import annotations

import numpy as np
import librosa
from functools import lru_cache
from typing import Optional

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock",
]

# Feature-vector index map (0-based, matching layout above)
_IDX = {
    "chroma_mean":    0,
    "chroma_var":     1,
    "rms_mean":       2,
    "rms_var":        3,
    "centroid_mean":  4,
    "centroid_var":   5,
    "bw_mean":        6,
    "bw_var":         7,
    "rolloff_mean":   8,
    "rolloff_var":    9,
    "zcr_mean":      10,
    "zcr_var":       11,
    "harmony_mean":  12,
    "harmony_var":   13,
    "perceptr_mean": 14,
    "perceptr_var":  15,
    "tempo":         16,
    # mfcc1..20 mean at 17..36, var at 37..56
}


# ── Feature extraction ──────────────────────────────────────────────────

def extract_genre_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute a 57-dimensional feature vector matching the GTZAN CSV format.
    This ensures compatibility with models trained on the GTZAN dataset.

    Returns ndarray of shape (57,), dtype float32.
    """
    hop = settings.hop_length

    # Harmonic / percussive separation
    y_harm, y_perc = librosa.effects.hpss(y)

    # ── Core spectral features ──────────────────────────────────────────
    chroma   = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    rms      = librosa.feature.rms(y=y, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    bw       = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]

    # ── Harmonic / percussive means ─────────────────────────────────────
    h_rms = librosa.feature.rms(y=y_harm, hop_length=hop)[0]
    p_rms = librosa.feature.rms(y=y_perc, hop_length=hop)[0]

    # ── Tempo ──────────────────────────────────────────────────────────
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    tempo_val = float(np.atleast_1d(tempo)[0])

    # ── MFCCs 1..20 ────────────────────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
    mfcc_mean = mfccs.mean(axis=1)   # (20,)
    mfcc_var  = mfccs.var(axis=1)    # (20,)

    features = np.array([
        # 0-1: chroma
        chroma.mean(), chroma.var(),
        # 2-3: rms
        rms.mean(), rms.var(),
        # 4-5: centroid
        centroid.mean(), centroid.var(),
        # 6-7: bandwidth
        bw.mean(), bw.var(),
        # 8-9: rolloff
        rolloff.mean(), rolloff.var(),
        # 10-11: zcr
        zcr.mean(), zcr.var(),
        # 12-13: harmony rms
        h_rms.mean(), h_rms.var(),
        # 14-15: percussive rms
        p_rms.mean(), p_rms.var(),
        # 16: tempo
        tempo_val,
        # 17-36: mfcc 1..20 mean
        *mfcc_mean,
        # 37-56: mfcc 1..20 var
        *mfcc_var,
    ], dtype=np.float32)   # shape (57,)

    return features


# ── Heuristic classifier ────────────────────────────────────────────────

# Reference centroids in the 4-D sub-space: [centroid, zcr, tempo, rms]
_GENRE_PROFILES = {
    "blues":     [1900, 0.060,  96, 0.110],
    "classical": [1600, 0.032,  80, 0.070],
    "country":   [2100, 0.075, 110, 0.120],
    "disco":     [2400, 0.085, 120, 0.150],
    "hiphop":    [1700, 0.055,  90, 0.130],
    "jazz":      [1800, 0.062,  95, 0.090],
    "metal":     [3200, 0.165, 145, 0.180],
    "pop":       [2300, 0.090, 118, 0.140],
    "reggae":    [1550, 0.060,  85, 0.115],
    "rock":      [2600, 0.130, 130, 0.160],
}
_SCALES = np.array([3000, 0.18, 180, 0.2], dtype=np.float32)


def _heuristic_classify(features: np.ndarray) -> np.ndarray:
    """
    Score each genre by Euclidean distance in a 4-D feature sub-space.
    Returns a softmax probability array of shape (10,).
    """
    centroid_mean = features[_IDX["centroid_mean"]]
    zcr_mean      = features[_IDX["zcr_mean"]]
    tempo         = features[_IDX["tempo"]]
    rms_mean      = features[_IDX["rms_mean"]]

    query = np.array([centroid_mean, zcr_mean, tempo, rms_mean], dtype=np.float32)
    query_norm = query / _SCALES

    scores = np.zeros(len(GENRES), dtype=np.float32)
    for i, genre in enumerate(GENRES):
        ref = np.array(_GENRE_PROFILES[genre], dtype=np.float32) / _SCALES
        dist = np.linalg.norm(query_norm - ref)
        scores[i] = 1.0 / (dist + 1e-6)

    exp_scores = np.exp(scores - scores.max())
    return exp_scores / exp_scores.sum()


# ── Model-based classifier ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_genre_model(model_path: str):
    """Load Keras genre model (cached)."""
    from tensorflow import keras
    return keras.models.load_model(model_path)


@lru_cache(maxsize=1)
def _load_genre_scaler(scaler_path: str):
    """Load sklearn scaler for genre features (cached)."""
    import joblib
    return joblib.load(scaler_path)


def _model_classify(features: np.ndarray) -> Optional[np.ndarray]:
    """
    Run genre model inference.  Returns probability array (10,) or None
    if the model / scaler can not be loaded.
    """
    try:
        model = _load_genre_model(str(settings.genre_model_path))
        try:
            scaler = _load_genre_scaler(str(settings.genre_scaler_path))
            x = scaler.transform(features.reshape(1, -1))
        except Exception:
            x = features.reshape(1, -1)
        probs = model.predict(x, verbose=0)[0]
        return probs.astype(np.float32)
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("Genre model inference failed (%s) — using heuristic.", exc)
        return None


# ── Public API ──────────────────────────────────────────────────────────

def classify_genre(y: np.ndarray, sr: int) -> dict:
    """
    Classify the genre of an audio signal.

    Returns::
        {
          "genre": "hiphop",
          "confidence": 0.82,
          "top3": [
              {"genre": "hiphop",  "confidence": 0.82},
              {"genre": "reggae",  "confidence": 0.10},
              {"genre": "blues",   "confidence": 0.05},
          ]
        }
    """
    features = extract_genre_features(y, sr)

    probs = _model_classify(features)
    if probs is None:
        logger.debug("Genre model not available — using heuristic.")
        probs = _heuristic_classify(features)

    top_idx = np.argsort(probs)[::-1]
    top3 = [
        {"genre": GENRES[i], "confidence": round(float(probs[i]), 4)}
        for i in top_idx[:3]
    ]
    return {
        "genre":      GENRES[top_idx[0]],
        "confidence": round(float(probs[top_idx[0]]), 4),
        "top3":       top3,
    }
