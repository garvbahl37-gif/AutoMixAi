"""
AutoMixAI – Request Schemas

Pydantic models for incoming API request bodies.
"""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request body for the ``POST /analyze`` endpoint."""

    file_id: str = Field(
        ...,
        description="Unique identifier of the previously uploaded audio file.",
        examples=["a1b2c3d4e5f6"],
    )


class MixRequest(BaseModel):
    """Request body for the ``POST /mix`` endpoint."""

    file_id_a: str = Field(
        ...,
        description="File ID of the first (outgoing) track.",
        examples=["a1b2c3d4e5f6"],
    )
    file_id_b: str = Field(
        ...,
        description="File ID of the second (incoming) track.",
        examples=["f6e5d4c3b2a1"],
    )
    crossfade_duration: float = Field(
        default=5.0,
        ge=0.5,
        le=30.0,
        description="Duration of the crossfade region in seconds.",
    )
