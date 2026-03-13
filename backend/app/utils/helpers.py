"""
AutoMixAI – Helper Utilities

Convenience functions used across multiple modules: unique ID generation,
path resolution, and startup directory creation.
"""

import uuid
from pathlib import Path

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def generate_file_id() -> str:
    """Generate a unique file identifier (UUID4 hex string)."""
    return uuid.uuid4().hex


def get_upload_path(file_id: str, extension: str = ".wav") -> Path:
    """
    Return the absolute storage path for an uploaded file.

    Args:
        file_id:   Unique identifier for the file.
        extension: File extension including the dot (e.g. ``".wav"``).

    Returns:
        Resolved :class:`Path` inside the uploads directory.
    """
    return settings.upload_dir / f"{file_id}{extension}"


def get_output_path(file_id: str, extension: str = ".wav") -> Path:
    """
    Return the absolute storage path for a generated output file.

    Args:
        file_id:   Unique identifier for the file.
        extension: File extension including the dot (e.g. ``".wav"``).

    Returns:
        Resolved :class:`Path` inside the outputs directory.
    """
    return settings.output_dir / f"{file_id}{extension}"


def ensure_directories() -> None:
    """
    Create all required storage and data directories if they do not
    already exist.  Called once at application startup.
    """
    directories = [
        settings.upload_dir,
        settings.output_dir,
        settings.model_dir,
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.labels_dir,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s", directory)
