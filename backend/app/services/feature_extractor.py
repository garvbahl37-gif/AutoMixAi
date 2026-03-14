"""
AutoMixAI – Feature Extractor Service

Extracts a 43-dimensional feature vector per audio frame for the v2 ANN
beat-detection model (trained on GTZAN + Hainsworth + Ballroom).

Feature layout (43 dims):
  [0:13]   MFCCs
  [13:26]  Delta-MFCCs     (velocity of MFCCs — captures beat impulse)
  [26:38]  Chroma STFT     (harmonic / key content)
  [38]     Onset strength
  [39]     Spectral flux
  [40]     RMS energy
  [41]     Spectral centroid
  [42]     Zero-crossing rate

Optimisations:
  - STFT computed ONCE and reused across all derived features.
  - INFERENCE_HOP = 1024 halves frame count vs training (512), keeping
    accuracy while cutting inference time roughly in half.
"""

import numpy as np
import librosa

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Hop length used during inference (larger → faster, near-identical results)
INFERENCE_HOP = 1024
FEATURE_DIM = 43   # must match the v2 training notebook


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract and concatenate all 43 features into a single matrix.

    Each row corresponds to one analysis frame.

    Args:
        y:  Audio time-series (mono, normalised).
        sr: Sample rate.

    Returns:
        Feature matrix of shape ``(n_frames, 43)``.
    """
    hop   = INFERENCE_HOP
    n_fft = settings.n_fft
    n_mfcc = settings.n_mfcc  # default 13

    # ── Single STFT pass ─────────────────────────────────────────────
    S_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    S_mag     = np.abs(S_complex)        # (n_fft//2+1, T)
    T         = S_mag.shape[1]

    # ── MFCCs (via mel-spectrogram, reuses S_mag) ────────────────────
    mel  = librosa.feature.melspectrogram(S=S_mag ** 2, sr=sr, n_fft=n_fft)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)  # (13, T)

    # ── Delta-MFCCs ──────────────────────────────────────────────────
    delta_mfcc = librosa.feature.delta(mfcc)                                  # (13, T)

    # ── Chroma (reuses S_mag) ────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(
        S=S_mag ** 2, sr=sr, hop_length=hop, n_fft=n_fft
    )                                                                           # (12, T)

    # ── Onset strength (db-scaled S_mag) ─────────────────────────────
    onset = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(S_mag, ref=np.max), sr=sr, hop_length=hop
    )                                                                           # (T,)

    # ── Spectral flux ─────────────────────────────────────────────────
    flux = np.sqrt(np.sum(np.diff(S_mag, axis=1) ** 2, axis=0))
    flux = np.concatenate([[0.0], flux])                                       # (T,)

    # ── RMS energy ───────────────────────────────────────────────────
    rms = librosa.feature.rms(S=S_mag, frame_length=n_fft, hop_length=hop)[0] # (T,)

    # ── Spectral centroid ─────────────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(
        S=S_mag, sr=sr, hop_length=hop, n_fft=n_fft
    )[0]                                                                        # (T,)

    # ── Zero-crossing rate ────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=n_fft, hop_length=hop
    )[0]                                                                        # (T,)

    # ── Align to common frame count & stack ──────────────────────────
    n_frames = min(
        T,
        mfcc.shape[1], delta_mfcc.shape[1], chroma.shape[1],
        len(onset), len(flux), len(rms), len(centroid), len(zcr),
    )

    features = np.column_stack([
        mfcc.T[:n_frames],         # 13
        delta_mfcc.T[:n_frames],   # 13
        chroma.T[:n_frames],       # 12
        onset[:n_frames],          # 1
        flux[:n_frames],           # 1
        rms[:n_frames],            # 1
        centroid[:n_frames],       # 1
        zcr[:n_frames],            # 1
    ])                             # → (n_frames, 43)

    logger.info(
        "Feature matrix: shape=%s  hop=%d  dim=%d",
        features.shape, hop, features.shape[1],
    )
    return features


# ── Legacy helpers (used by training pipeline) ────────────────────────

def extract_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=settings.n_mfcc,
        hop_length=settings.hop_length, n_fft=settings.n_fft,
    ).T


def extract_onset_strength(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=settings.hop_length
    )


def extract_spectral_flux(y: np.ndarray, sr: int) -> np.ndarray:
    S = np.abs(librosa.stft(y, n_fft=settings.n_fft, hop_length=settings.hop_length))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    return np.concatenate([[0.0], flux])


def extract_energy_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.rms(
        y=y, frame_length=settings.n_fft, hop_length=settings.hop_length
    )[0]
