"""
AutoMixAI – Dataset Preparation Script

Reads audio files and annotations from the Ballroom and FMA datasets
in the ``Datasets/`` folder, extracts features, and writes processed
training data to ``data/processed/``.

Supported datasets:
    • **Ballroom** – 698 WAV files organised by genre with ``.bpm``
      ground-truth annotation files.
    • **FMA Small** – ~8 000 MP3 tracks with metadata in
      ``fma_metadata/tracks.csv``.

Usage::

    cd backend
    python -m app.data.prepare_data
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np

# ── Resolve project paths ────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent               # app/data/
_APP_DIR = _SCRIPT_DIR.parent                               # app/
_BACKEND_DIR = _APP_DIR.parent                              # backend/
_PROJECT_DIR = _BACKEND_DIR.parent                          # AutoMixAI/

sys.path.insert(0, str(_BACKEND_DIR))

from app.services.audio_loader import load_audio
from app.services.feature_extractor import extract_features
from app.services.beat_detector import detect_beats_librosa
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Dataset paths ────────────────────────────────────────────────────
DATASETS_DIR = _PROJECT_DIR / "Datasets"
BALLROOM_AUDIO_DIR = DATASETS_DIR / "BallroomData"
BALLROOM_BPM_DIR = DATASETS_DIR / "BallroomAnnotations" / "ballroomGroundTruth"
FMA_AUDIO_DIR = DATASETS_DIR / "fma_small" / "fma_small"
FMA_METADATA_DIR = DATASETS_DIR / "fma_metadata" / "fma_metadata"

# Output
PROCESSED_DIR = _BACKEND_DIR / "data" / "processed"


# ── Ballroom dataset ─────────────────────────────────────────────────

def load_ballroom_bpm_annotations() -> dict[str, float]:
    """
    Read all ``.bpm`` files from the Ballroom annotations directory.

    Returns:
        Dict mapping stem name (e.g. ``"Albums-Ballroom_Classics4-01"``)
        to its ground-truth BPM value.
    """
    bpm_map: dict[str, float] = {}
    if not BALLROOM_BPM_DIR.exists():
        logger.warning("Ballroom annotations dir not found: %s", BALLROOM_BPM_DIR)
        return bpm_map

    for bpm_file in BALLROOM_BPM_DIR.glob("*.bpm"):
        try:
            bpm_val = float(bpm_file.read_text().strip())
            bpm_map[bpm_file.stem] = bpm_val
        except ValueError:
            logger.warning("Skipping invalid .bpm file: %s", bpm_file.name)

    logger.info("Loaded %d Ballroom BPM annotations", len(bpm_map))
    return bpm_map


def prepare_ballroom(max_files: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Process Ballroom WAV files → (features, labels).

    For each file:
        1. Load and normalise audio.
        2. Extract frame-level features.
        3. Generate pseudo beat labels via librosa.

    The ground-truth BPM from ``.bpm`` files is stored in a side CSV
    for reference but is **not** used as the label (we need per-frame
    beat/non-beat labels).

    Args:
        max_files: Optional cap on the number of files to process
                   (useful for quick testing).

    Returns:
        ``(X, y)`` arrays.
    """
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    bpm_map = load_ballroom_bpm_annotations()

    wav_files = sorted(BALLROOM_AUDIO_DIR.rglob("*.wav"))
    if max_files:
        wav_files = wav_files[:max_files]

    logger.info("Processing %d Ballroom files …", len(wav_files))

    for i, wav_path in enumerate(wav_files, 1):
        try:
            y_audio, sr = load_audio(str(wav_path))
            features = extract_features(y_audio, sr)

            # Generate per-frame beat labels using librosa beat tracker
            beat_times = detect_beats_librosa(y_audio, sr)
            n_frames = features.shape[0]

            labels = np.zeros(n_frames, dtype=np.float32)
            import librosa as _lr
            beat_frames = _lr.time_to_frames(
                beat_times, sr=sr, hop_length=settings.hop_length
            )
            for bf in beat_frames:
                if 0 <= bf < n_frames:
                    labels[bf] = 1.0

            all_features.append(features)
            all_labels.append(labels)

            if i % 50 == 0:
                logger.info("  Ballroom progress: %d / %d", i, len(wav_files))

        except Exception as exc:
            logger.warning("Skipping %s: %s", wav_path.name, exc)

    if not all_features:
        return np.array([]), np.array([])

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    logger.info("Ballroom dataset: X=%s  y=%s  (beat ratio=%.3f)", X.shape, y.shape, y.mean())
    return X, y


# ── FMA Small dataset ────────────────────────────────────────────────

def load_fma_track_ids(subset: str = "small") -> list[int]:
    """
    Read ``tracks.csv`` and return track IDs belonging to *subset*.

    Args:
        subset: One of ``"small"``, ``"medium"``, ``"large"``, ``"full"``.

    Returns:
        List of integer track IDs.
    """
    tracks_csv = FMA_METADATA_DIR / "tracks.csv"
    if not tracks_csv.exists():
        logger.warning("FMA tracks.csv not found at %s", tracks_csv)
        return []

    track_ids: list[int] = []

    with open(tracks_csv, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header_row_1 = next(reader)     # top-level header
        header_row_2 = next(reader)     # sub-level header
        header_row_3 = next(reader)     # column types (skip)

        # Find the column index for 'set' → 'subset'
        # The FMA CSV uses a multi-row header.  Column 0 is track_id,
        # and the 'subset' field is typically at a known offset.
        # We'll scan for it.
        subset_col = None
        for idx, (top, sub) in enumerate(zip(header_row_1, header_row_2)):
            if top.strip().lower() == "set" and sub.strip().lower() == "subset":
                subset_col = idx
                break

        if subset_col is None:
            # Fallback: search just in header_row_2
            for idx, sub in enumerate(header_row_2):
                if sub.strip().lower() == "subset":
                    subset_col = idx
                    break

        if subset_col is None:
            logger.warning("Could not find 'subset' column in tracks.csv")
            return []

        for row in reader:
            try:
                if row[subset_col].strip().lower() == subset:
                    track_ids.append(int(row[0]))
            except (IndexError, ValueError):
                continue

    logger.info("Found %d FMA '%s' track IDs", len(track_ids), subset)
    return track_ids


def _fma_track_path(track_id: int) -> Path:
    """Return the expected path for an FMA track MP3 file."""
    # FMA stores files as: fma_small/000/000002.mp3
    tid = f"{track_id:06d}"
    return FMA_AUDIO_DIR / tid[:3] / f"{tid}.mp3"


def prepare_fma(max_files: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Process FMA Small MP3 files → (features, labels).

    Beat labels are pseudo-generated via librosa since FMA does not
    include beat annotations.

    Args:
        max_files: Optional cap for testing.

    Returns:
        ``(X, y)`` arrays.
    """
    track_ids = load_fma_track_ids("small")
    if max_files:
        track_ids = track_ids[:max_files]

    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    logger.info("Processing %d FMA Small files …", len(track_ids))

    for i, tid in enumerate(track_ids, 1):
        mp3_path = _fma_track_path(tid)
        if not mp3_path.exists():
            continue

        try:
            y_audio, sr = load_audio(str(mp3_path))
            features = extract_features(y_audio, sr)

            beat_times = detect_beats_librosa(y_audio, sr)
            n_frames = features.shape[0]

            labels = np.zeros(n_frames, dtype=np.float32)
            import librosa as _lr
            beat_frames = _lr.time_to_frames(
                beat_times, sr=sr, hop_length=settings.hop_length
            )
            for bf in beat_frames:
                if 0 <= bf < n_frames:
                    labels[bf] = 1.0

            all_features.append(features)
            all_labels.append(labels)

            if i % 100 == 0:
                logger.info("  FMA progress: %d / %d", i, len(track_ids))

        except Exception as exc:
            logger.warning("Skipping FMA track %06d: %s", tid, exc)

    if not all_features:
        return np.array([]), np.array([])

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    logger.info("FMA dataset: X=%s  y=%s  (beat ratio=%.3f)", X.shape, y.shape, y.mean())
    return X, y


# ── Combined preparation ─────────────────────────────────────────────

def prepare_all(
    max_ballroom: int | None = None,
    max_fma: int | None = None,
    save: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process both datasets and optionally save to ``data/processed/``.

    Args:
        max_ballroom: Limit Ballroom files (for testing).
        max_fma:      Limit FMA files (for testing).
        save:         If ``True``, save ``X.npy`` and ``y.npy``.

    Returns:
        Combined ``(X, y)`` training arrays.
    """
    parts_X: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []

    # ── Ballroom ──────────────────────────────────────────────────────
    if BALLROOM_AUDIO_DIR.exists():
        X_b, y_b = prepare_ballroom(max_ballroom)
        if len(X_b) > 0:
            parts_X.append(X_b)
            parts_y.append(y_b)
    else:
        logger.info("Ballroom dataset not found — skipping")

    # ── FMA Small ─────────────────────────────────────────────────────
    if FMA_AUDIO_DIR.exists():
        X_f, y_f = prepare_fma(max_fma)
        if len(X_f) > 0:
            parts_X.append(X_f)
            parts_y.append(y_f)
    else:
        logger.info("FMA Small dataset not found — skipping")

    if not parts_X:
        raise RuntimeError("No data found. Check that datasets are in Datasets/")

    X = np.vstack(parts_X)
    y = np.concatenate(parts_y)
    logger.info("Combined dataset: X=%s  y=%s", X.shape, y.shape)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(PROCESSED_DIR / "X.npy"), X)
        np.save(str(PROCESSED_DIR / "y.npy"), y)
        logger.info("Saved processed data to %s", PROCESSED_DIR)

    return X, y


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare training data for AutoMixAI")
    parser.add_argument("--max-ballroom", type=int, default=None,
                        help="Max Ballroom files to process (for testing)")
    parser.add_argument("--max-fma", type=int, default=None,
                        help="Max FMA files to process (for testing)")
    args = parser.parse_args()

    prepare_all(max_ballroom=args.max_ballroom, max_fma=args.max_fma)
