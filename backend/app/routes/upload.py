"""
AutoMixAI – Upload Route

``POST /upload``  Accept an audio file, persist it to the uploads
directory, and return a unique file ID for subsequent operations.
"""

from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.analysis_response import UploadResponse
from app.services.audio_loader import get_audio_info
from app.utils.helpers import generate_file_id, get_upload_path
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Upload"])

# Accepted MIME prefixes / extensions
_ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


@router.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an audio file for analysis or mixing.

    The file is saved under a unique ID in ``storage/uploads/``.
    """
    # ── Validate extension ────────────────────────────────────────────
    original_name = file.filename or "unknown"
    ext = Path(original_name).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
            ),
        )

    # ── Save to disk ──────────────────────────────────────────────────
    file_id = generate_file_id()
    save_path = get_upload_path(file_id, ext)

    logger.info("Saving upload '%s' → %s", original_name, save_path)

    try:
        contents = await file.read()
        save_path.write_bytes(contents)
    except Exception as exc:
        logger.error("Failed to save upload: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save file.") from exc

    # ── Retrieve basic audio metadata ─────────────────────────────────
    try:
        info = get_audio_info(str(save_path))
    except Exception as exc:
        logger.error("Uploaded file is not a valid audio file: %s", exc)
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid or supported audio file.",
        ) from exc

    return UploadResponse(
        file_id=file_id,
        filename=original_name,
        duration=round(info["duration"], 2),
    )
