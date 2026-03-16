"""
AutoMixAI – Beat Generation Request/Response Schemas
"""

from pydantic import BaseModel, Field


class GenerateBeatRequest(BaseModel):
    """Request body for POST /generate."""

    prompt: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural-language description of the beat to generate.",
        examples=["chill hip-hop beat at 90 BPM with heavy bass"],
    )
    bars: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of bars to generate (overrides prompt if provided).",
    )


class PatternInfo(BaseModel):
    kick:    list[int]
    snare:   list[int]
    hihat_c: list[int]
    hihat_o: list[int]
    clap:    list[int]


class GenerateBeatResponse(BaseModel):
    """Response returned after beat generation."""

    output_file_id: str = Field(..., description="File ID — use GET /output/{id} to download.")
    genre:          str   = Field(..., description="Detected / selected genre.")
    bpm:            float = Field(..., description="Beats per minute used.")
    bars:           int   = Field(..., description="Number of bars rendered.")
    complexity:     str   = Field(..., description="Pattern complexity level.")
    description:    str   = Field(..., description="Human-readable summary of the generated beat.")
    duration:       float = Field(..., description="Duration of the generated audio in seconds.")
    pattern:        PatternInfo = Field(..., description="16-step drum pattern that was rendered.")
    sample_rate:    int   = Field(default=44100, description="Sample rate of the output WAV.")
    message:        str   = Field(default="Beat generated successfully.")
