"""
AutoMixAI – Application Configuration

Centralizes all configurable parameters: file paths, audio processing
settings, ANN model hyperparameters, and server options.  Values are
loaded from environment variables (with sensible defaults) using
pydantic-settings.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


# ── Project root resolved relative to this file ──────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent          # backend/app/
_BACKEND_DIR = _APP_DIR.parent                             # backend/


class Settings(BaseSettings):
    """Global configuration for AutoMixAI."""

    # ── Storage paths ─────────────────────────────────────────────────
    upload_dir: Path = _APP_DIR / "storage" / "uploads"
    output_dir: Path = _APP_DIR / "storage" / "outputs"
    model_dir: Path = _APP_DIR / "storage" / "models"

    # ── Training data paths ───────────────────────────────────────────
    raw_data_dir: Path = _BACKEND_DIR / "data" / "raw"
    processed_data_dir: Path = _BACKEND_DIR / "data" / "processed"
    labels_dir: Path = _BACKEND_DIR / "data" / "labels"

    # ── Audio processing parameters ───────────────────────────────────
    sample_rate: int = 22050
    hop_length: int = 512
    n_fft: int = 2048
    n_mfcc: int = 13

    # ── ANN model parameters ─────────────────────────────────────────
    beat_threshold: float = 0.5
    model_filename: str = "beat_detector.h5"
    training_epochs: int = 50
    training_batch_size: int = 32

    # ── Mixing defaults ──────────────────────────────────────────────
    default_crossfade_duration: float = 5.0   # seconds

    # ── Server ────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    @property
    def model_path(self) -> Path:
        """Full path to the trained ANN model file."""
        return self.model_dir / self.model_filename

    class Config:
        env_prefix = "AUTOMIX_"


# Singleton instance used throughout the application
settings = Settings()
