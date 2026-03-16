"""
AutoMixAI – Audio Loader Service

Responsible for loading audio files from disk, converting to mono,
normalizing amplitude, and writing processed audio back to disk.
"""

import numpy as np
import librosa
import soundfile as sf

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_audio(
    path: str,
    sr: int | None = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file from *path*, optionally resample, and normalize.

    Args:
        path: Absolute or relative path to the audio file.
        sr:   Target sample rate.  Defaults to ``settings.sample_rate``.
        mono: If ``True``, down-mix to a single channel.

    Returns:
        Tuple of ``(audio_signal, sample_rate)``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError:      If the file cannot be decoded.
    """
    sr = sr or settings.sample_rate

    logger.info("Loading audio: %s  (target sr=%d, mono=%s)", path, sr, mono)

    try:
        y, loaded_sr = librosa.load(path, sr=sr, mono=mono)
    except FileNotFoundError:
        logger.error("Audio file not found: %s", path)
        raise
    except Exception as exc:
        logger.error("Failed to decode audio file: %s – %s", path, exc)
        raise RuntimeError(f"Cannot decode audio file: {path}") from exc

    # Normalize amplitude to [-1, 1]
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
        logger.debug("Normalized audio (peak was %.4f)", peak)

    logger.info(
        "Loaded audio: %.2f s, %d samples, sr=%d",
        len(y) / loaded_sr,
        len(y),
        loaded_sr,
    )
    return y, loaded_sr


def save_audio(path: str, audio: np.ndarray, sr: int) -> None:
    """
    Write an audio array to disk as a WAV file.

    Args:
        path:  Destination path (including extension).
        audio: 1-D NumPy array of audio samples.
        sr:    Sample rate.
    """
    logger.info("Saving audio to %s (sr=%d, samples=%d)", path, sr, len(audio))
    sf.write(path, audio, sr)


def get_audio_info(path: str) -> dict:
    """
    Return basic metadata for an audio file without loading all samples.

    Args:
        path: Path to the audio file.

    Returns:
        Dictionary with keys ``duration``, ``sample_rate``, ``channels``.
    """
    try:
        # Fast path: soundfile handles WAV/FLAC natively
        info = sf.info(path)
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
        }
    except Exception:
        # Fallback for MP3/AAC/M4A/OGG (soundfile doesn't support these)
        y, sr = librosa.load(path, sr=None, mono=False)
        channels = 1 if y.ndim == 1 else y.shape[0]
        duration = (len(y) if y.ndim == 1 else y.shape[1]) / sr
        return {
            "duration": float(duration),
            "sample_rate": int(sr),
            "channels": int(channels),
        }
