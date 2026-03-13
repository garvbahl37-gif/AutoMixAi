"""
AutoMixAI – Analyze Route

``POST /analyze``  Run beat detection and BPM estimation on a
previously uploaded audio file.

The CPU-intensive work (feature extraction + ANN inference) is
offloaded to a thread-pool executor so it never blocks the async
event loop.
"""

import asyncio
import time
from functools import partial
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.schemas.audio_request import AnalyzeRequest
from app.schemas.analysis_response import AnalysisResponse
from app.services.audio_loader import load_audio
from app.services.beat_detector import detect_beats
from app.services.bpm_estimator import estimate_bpm_from_beats, estimate_bpm_librosa
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Analysis"])


def _find_upload(file_id: str) -> Path:
    """Locate the uploaded file by scanning the uploads directory."""
    for path in settings.upload_dir.iterdir():
        if path.stem == file_id:
            return path
    raise FileNotFoundError(f"No uploaded file found for ID '{file_id}'")


def _run_analysis(file_path: Path):
    """
    Synchronous analysis pipeline — runs in a thread pool so it
    doesn't block the event loop.

    Returns (bpm, beat_times, duration, sr).
    """
    t0 = time.perf_counter()

    y, sr = load_audio(str(file_path))
    t1 = time.perf_counter()
    logger.info("Audio loaded in %.2f s", t1 - t0)

    beat_times = detect_beats(y, sr)
    t2 = time.perf_counter()
    logger.info("Beat detection in %.2f s  (%d beats)", t2 - t1, len(beat_times))

    bpm = (
        estimate_bpm_from_beats(beat_times)
        if len(beat_times) >= 2
        else estimate_bpm_librosa(y, sr)
    )
    t3 = time.perf_counter()
    logger.info("BPM estimation in %.2f s  (BPM=%.2f)", t3 - t2, bpm)
    logger.info("Total analysis time: %.2f s", t3 - t0)

    return bpm, beat_times, round(len(y) / sr, 2), sr


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(request: AnalyzeRequest):
    """
    Analyze an uploaded audio file: detect beats and estimate BPM.
    Heavy CPU work runs in a thread pool to keep the server responsive.
    """
    logger.info("Analyze request for file_id=%s", request.file_id)

    # ── Locate the file ───────────────────────────────────────────────
    try:
        file_path = _find_upload(request.file_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"File with ID '{request.file_id}' not found.",
        )

    # ── Run blocking pipeline in thread pool ─────────────────────────
    try:
        loop = asyncio.get_event_loop()
        bpm, beat_times, duration, sr = await loop.run_in_executor(
            None, partial(_run_analysis, file_path)
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AnalysisResponse(
        file_id=request.file_id,
        bpm=bpm,
        beat_times=beat_times,
        duration=duration,
        sample_rate=sr,
    )
