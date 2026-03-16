"""
AutoMixAI – Beat Generation Route

POST /generate   Synthesise a drum beat from a natural-language prompt.
                 CPU-intensive rendering runs in a thread-pool executor.
"""

import asyncio
from functools import partial

from fastapi import APIRouter, HTTPException

from app.schemas.generate_request import GenerateBeatRequest, GenerateBeatResponse, PatternInfo
from app.services.beat_generator import generate_beat
from app.utils.helpers import generate_file_id, get_output_path
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Generate"])


def _run_generation(prompt: str, bars: int, output_path: str) -> dict:
    """Synchronous generation pipeline — runs in a thread pool."""
    # If caller passed an explicit bars override, append it to the prompt
    # so parse_prompt() picks it up.
    effective_prompt = prompt
    if bars:
        effective_prompt = f"{prompt} {bars} bars"
    return generate_beat(effective_prompt, output_path)


@router.post("/generate", response_model=GenerateBeatResponse)
async def generate_beat_route(request: GenerateBeatRequest):
    """
    Generate a synthesised drum beat from a text description.
    The audio is saved and available via GET /output/{output_file_id}.
    """
    logger.info("Generate request: prompt=%r bars=%d", request.prompt, request.bars)

    output_id   = generate_file_id()
    output_path = get_output_path(output_id, extension=".wav")

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(_run_generation, request.prompt, request.bars, str(output_path)),
        )
    except Exception as exc:
        logger.error("Beat generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Beat generation failed: {exc}") from exc

    pattern_raw = result["pattern"]
    return GenerateBeatResponse(
        output_file_id = output_id,
        genre          = result["genre"],
        bpm            = result["bpm"],
        bars           = result["bars"],
        complexity     = result["complexity"],
        description    = result["description"],
        duration       = result["duration"],
        sample_rate    = result["sample_rate"],
        pattern        = PatternInfo(
            kick    = pattern_raw.get("kick",    [0]*16),
            snare   = pattern_raw.get("snare",   [0]*16),
            hihat_c = pattern_raw.get("hihat_c", [0]*16),
            hihat_o = pattern_raw.get("hihat_o", [0]*16),
            clap    = pattern_raw.get("clap",    [0]*16),
        ),
    )
