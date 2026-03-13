"""
AutoMixAI – Response Schemas

Pydantic models for API response bodies.
"""

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response returned after a successful file upload."""

    file_id: str = Field(..., description="Unique identifier assigned to the upload.")
    filename: str = Field(..., description="Original filename of the uploaded file.")
    duration: float = Field(..., description="Duration of the audio in seconds.")
    message: str = Field(
        default="File uploaded successfully.",
        description="Human-readable status message.",
    )


class AnalysisResponse(BaseModel):
    """Response returned after audio analysis (BPM + beat detection)."""

    file_id: str = Field(..., description="Identifier of the analyzed file.")
    bpm: float = Field(..., description="Estimated tempo in beats per minute.")
    beat_times: list[float] = Field(
        ..., description="Timestamps (in seconds) of detected beats."
    )
    duration: float = Field(..., description="Total duration of the audio in seconds.")
    sample_rate: int = Field(..., description="Sample rate used during analysis.")
    message: str = Field(
        default="Analysis complete.",
        description="Human-readable status message.",
    )


class MixResponse(BaseModel):
    """Response returned after generating a DJ mix."""

    output_file_id: str = Field(
        ..., description="File ID of the generated mix (use with GET /output/{file_id})."
    )
    duration: float = Field(..., description="Duration of the mix in seconds.")
    bpm_a: float = Field(..., description="Detected BPM of track A.")
    bpm_b: float = Field(..., description="Detected BPM of track B.")
    target_bpm: float = Field(..., description="BPM used for the synchronized mix.")
    message: str = Field(
        default="Mix generated successfully.",
        description="Human-readable status message.",
    )
