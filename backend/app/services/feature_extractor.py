"""
AutoMixAI – Feature Extractor Service

Extracts audio features (MFCCs, onset strength, spectral flux, energy
envelope) and combines them into a feature matrix suitable for the ANN
beat-detection model.

Optimised for speed:
- The STFT magnitude spectrum is computed **once** and reused across all
  feature functions, avoiding redundant FFT passes.
- An ``inference_hop`` parameter (default 1 024) reduces the frame count
  by ~2× compared with the training hop of 512, with negligible loss in
  beat-detection accuracy.
"""

import numpy as np
import librosa

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Hop length used specifically during inference (larger = faster, ~same accuracy)
INFERENCE_HOP = 1024


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract and concatenate all features into a single matrix.

    Each row corresponds to one analysis frame.  Columns are ordered as:
    ``[MFCCs (n_mfcc) | onset_strength (1) | spectral_flux (1) | energy (1)]``

    The STFT is computed once and shared between all derived features.

    Args:
        y:  Audio time-series (mono, normalized).
        sr: Sample rate.

    Returns:
        Feature matrix of shape ``(n_frames, n_mfcc + 3)``.
    """
    hop = INFERENCE_HOP
    n_fft = settings.n_fft
    n_mfcc = settings.n_mfcc

    # ── Single STFT pass ─────────────────────────────────────────────────
    S_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    S_mag = np.abs(S_complex)                    # (n_fft//2+1, n_frames)
    n_frames = S_mag.shape[1]

    # ── MFCCs (reuse S_mag via mel spectrogram) ───────────────────────────
    mel = librosa.feature.melspectrogram(S=S_mag ** 2, sr=sr, n_fft=n_fft)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)
    mfcc = mfcc.T[:n_frames]                     # (n_frames, n_mfcc)

    # ── Onset strength (reuse S_mag) ──────────────────────────────────────
    onset = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(S_mag, ref=np.max),
        sr=sr,
        hop_length=hop,
    )[:n_frames]                                  # (n_frames,)

    # ── Spectral flux (column-difference of S_mag) ────────────────────────
    flux = np.sqrt(np.sum(np.diff(S_mag, axis=1) ** 2, axis=0))
    flux = np.concatenate([[0.0], flux])[:n_frames]  # (n_frames,)

    # ── RMS energy envelope ───────────────────────────────────────────────
    rms = librosa.feature.rms(
        S=S_mag, frame_length=n_fft, hop_length=hop
    )[0][:n_frames]                               # (n_frames,)

    features = np.column_stack([mfcc, onset, flux, rms])
    logger.info(
        "Feature matrix: shape=%s  hop=%d  (~%.1f s per frame)",
        features.shape, hop, hop / sr,
    )
    return features


# ── Kept for backwards-compatibility (training pipeline still uses these) ──

def extract_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=settings.n_mfcc,
        hop_length=settings.hop_length, n_fft=settings.n_fft,
    )
    return mfcc.T


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
