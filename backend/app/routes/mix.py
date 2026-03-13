"""
AutoMixAI – Mix & Output Routes

``POST /mix``             Generate a beat-synchronized DJ mix from two
                          uploaded tracks.
``GET  /output/{file_id}``  Download a previously generated mix.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.schemas.audio_request import MixRequest
from app.schemas.analysis_response import MixResponse
from app.services.mixer import create_mix
from app.utils.config import settings
from app.utils.helpers import generate_file_id, get_output_path
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Mixing"])


def _find_upload(file_id: str) -> Path:
    """Locate the uploaded file by scanning the uploads directory."""
    for path in settings.upload_dir.iterdir():
        if path.stem == file_id:
            return path
    raise FileNotFoundError(f"No uploaded file found for ID '{file_id}'")


# ── POST /mix ─────────────────────────────────────────────────────────

@router.post("/mix", response_model=MixResponse)
async def mix_tracks(request: MixRequest):
    """
    Generate a beat-synchronized mix from two uploaded audio tracks.

    The mix is saved to ``storage/outputs/`` and can be retrieved with
    ``GET /output/{output_file_id}``.
    """
    logger.info(
        "Mix request: A=%s, B=%s, crossfade=%.1f s",
        request.file_id_a,
        request.file_id_b,
        request.crossfade_duration,
    )

    # ── Resolve source files ──────────────────────────────────────────
    try:
        path_a = _find_upload(request.file_id_a)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Track A (ID '{request.file_id_a}') not found.",
        )

    try:
        path_b = _find_upload(request.file_id_b)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Track B (ID '{request.file_id_b}') not found.",
        )

    # ── Run mixing pipeline ───────────────────────────────────────────
    output_id = generate_file_id()
    output_path = get_output_path(output_id, extension=".wav")

    try:
        result = create_mix(
            path_a=str(path_a),
            path_b=str(path_b),
            output_path=str(output_path),
            crossfade_duration=request.crossfade_duration,
        )
    except ValueError as exc:
        logger.error("Mix failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected mixing error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during mixing.",
        ) from exc

    return MixResponse(
        output_file_id=output_id,
        duration=result["duration"],
        bpm_a=result["bpm_a"],
        bpm_b=result["bpm_b"],
        target_bpm=result["target_bpm"],
    )


# ── GET /output/{file_id} ────────────────────────────────────────────

@router.get("/output/{file_id}")
async def download_output(file_id: str):
    """
    Download a previously generated mix by its file ID.
    """
    output_path = get_output_path(file_id, extension=".wav")

    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Output file '{file_id}' not found.",
        )

    logger.info("Serving output file: %s", output_path)
    return FileResponse(
        path=str(output_path),
        media_type="audio/wav",
        filename=f"automix_{file_id}.wav",
    )
