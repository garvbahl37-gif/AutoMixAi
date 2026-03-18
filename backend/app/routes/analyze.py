"""
AutoMixAI – Analyze Route

``POST /analyze``  Run comprehensive audio analysis:
- Beat detection and BPM estimation
- Genre classification (GTZAN model)
- Music tag prediction (MagnaTagATune model)
- Instrument classification (NSynth model)
- Energy and mood analysis

The CPU-intensive work is offloaded to a thread-pool executor.
"""

import asyncio
import time
from functools import partial
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.schemas.audio_request import AnalyzeRequest
from app.schemas.analysis_response import (
    AnalysisResponse, GenreTopResult, TagScore, InstrumentResult
)
from app.services.audio_loader import load_audio
from app.services.beat_detector import detect_beats
from app.services.bpm_estimator import estimate_bpm_from_beats, estimate_bpm_librosa
from app.services.genre_classifier import classify_genre
from app.services.tag_predictor import predict_tags
from app.services.instrument_classifier import classify_instruments
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

    Returns comprehensive analysis results.
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

    genre_result = classify_genre(y, sr)
    t4 = time.perf_counter()
    logger.info(
        "Genre classification in %.2f s  (genre=%s, conf=%.2f)",
        t4 - t3, genre_result["genre"], genre_result["confidence"],
    )

    # ── New: Tag prediction ───────────────────────────────────────────
    tag_result = predict_tags(y, sr)
    t5 = time.perf_counter()
    logger.info(
        "Tag prediction in %.2f s  (%d tags, mood=%s)",
        t5 - t4, len(tag_result.get("tags", [])), tag_result.get("mood", "unknown"),
    )

    # ── New: Instrument classification ────────────────────────────────
    inst_result = classify_instruments(y, sr)
    t6 = time.perf_counter()
    logger.info(
        "Instrument classification in %.2f s  (instrument=%s, conf=%.2f)",
        t6 - t5, inst_result.get("instrument", "unknown"), inst_result.get("confidence", 0),
    )

    # ── Derive energy level ───────────────────────────────────────────
    import librosa
    rms = librosa.feature.rms(y=y).mean()
    if rms > 0.12:
        energy = "high"
    elif rms > 0.06:
        energy = "medium"
    else:
        energy = "low"

    logger.info("Total analysis time: %.2f s", t6 - t0)

    return {
        "bpm": bpm,
        "beat_times": beat_times,
        "duration": round(len(y) / sr, 2),
        "sr": sr,
        "genre_result": genre_result,
        "tag_result": tag_result,
        "inst_result": inst_result,
        "energy": energy,
    }


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(request: AnalyzeRequest):
    """
    Analyze an uploaded audio file comprehensively:
    - Detect beats and estimate BPM
    - Classify genre (10 classes)
    - Predict music tags (instruments, mood, style)
    - Identify dominant instrument family
    - Analyze energy level

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
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, partial(_run_analysis, file_path)
        )
    except Exception as exc:
        logger.error("Analysis failed for %s: %s", request.file_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    # ── Build response ────────────────────────────────────────────────
    genre_result = result["genre_result"]
    tag_result = result["tag_result"]
    inst_result = result["inst_result"]

    return AnalysisResponse(
        file_id=request.file_id,
        bpm=result["bpm"],
        beat_times=result["beat_times"],
        duration=result["duration"],
        sample_rate=result["sr"],
        # Genre
        genre=genre_result["genre"],
        genre_confidence=genre_result["confidence"],
        genre_top3=[GenreTopResult(**g) for g in genre_result["top3"]],
        # Tags
        tags=tag_result.get("tags", []),
        tag_scores=[TagScore(**t) for t in tag_result.get("tag_scores", [])],
        mood=tag_result.get("mood", "neutral"),
        has_vocals=tag_result.get("has_vocals", False),
        # Instruments
        dominant_instrument=inst_result.get("instrument", "unknown"),
        instrument_confidence=inst_result.get("confidence", 0.0),
        instruments_top3=[InstrumentResult(**i) for i in inst_result.get("top_k", [])],
        # Energy
        energy=result["energy"],
    )
