"""
AutoMixAI Beat Generator — HuggingFace Space

AI-powered music/beat generation using Meta's MusicGen model.
Generates studio-quality beats, loops, and music from text prompts.

Endpoints:
  POST /generate   Generate beat/music from text prompt
  GET  /output/{id} Download generated audio
  GET  /health     Health check
"""

import os
import uuid
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path(tempfile.gettempdir()) / "automixai_beats"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model selection: small for speed, medium for quality
MODEL_ID = os.environ.get("MUSICGEN_MODEL", "facebook/musicgen-small")
SAMPLE_RATE = 32000  # MusicGen outputs at 32kHz

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AutoMixAI Beat Generator",
    description="AI-powered beat/music generation using Meta's MusicGen.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500,
        description="Text prompt describing the beat/music to generate")
    duration: int = Field(default=10, ge=3, le=30,
        description="Duration in seconds (3-30)")
    temperature: float = Field(default=1.0, ge=0.5, le=1.5,
        description="Generation temperature: lower=more predictable, higher=more creative")
    guidance_scale: float = Field(default=3.0, ge=1.0, le=10.0,
        description="How closely to follow the prompt (higher=stricter)")

class GenerateResponse(BaseModel):
    output_file_id: str
    prompt: str
    duration: float
    model: str
    sample_rate: int
    message: str = "Beat generated successfully."

# ═══════════════════════════════════════════════════════════════════════════════
# MUSICGEN MODEL (Lazy-loaded)
# ═══════════════════════════════════════════════════════════════════════════════

_model = None
_processor = None

def _load_model():
    """Lazy-load the MusicGen model and processor."""
    global _model, _processor
    if _model is None:
        print(f"Loading MusicGen model: {MODEL_ID}")
        start = time.time()

        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        if device == "cuda":
            _model = _model.half()  # FP16 for faster GPU inference

        elapsed = time.time() - start
        print(f"MusicGen loaded on {device} in {elapsed:.1f}s")

    return _model, _processor


def generate_music(prompt: str, duration: int = 10, temperature: float = 1.0,
                   guidance_scale: float = 3.0) -> tuple:
    """
    Generate music/beat from text prompt using MusicGen.

    Returns (audio_array, sample_rate)
    """
    model, processor = _load_model()
    device = next(model.parameters()).device

    # Process the prompt
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Calculate max_new_tokens from duration
    # MusicGen generates at ~50 tokens/second at 32kHz
    tokens_per_second = 50
    max_new_tokens = int(duration * tokens_per_second)

    print(f"Generating: '{prompt}' ({duration}s, temp={temperature}, guidance={guidance_scale})")
    start = time.time()

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            guidance_scale=guidance_scale,
            do_sample=True,
        )

    elapsed = time.time() - start
    print(f"Generation complete in {elapsed:.1f}s")

    # Convert to numpy
    audio = audio_values[0, 0].cpu().numpy()

    # Normalize to prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    return audio, SAMPLE_RATE


# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "AutoMixAI Beat Generator v1.0",
        "model": MODEL_ID,
        "features": ["text-to-music", "text-to-beat"],
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_ID}


@app.post("/generate", response_model=GenerateResponse)
async def generate_beat(request: GenerateRequest):
    """Generate a beat/music clip from a text prompt using MusicGen."""
    output_id = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"{output_id}.wav"

    try:
        audio, sr = generate_music(
            prompt=request.prompt,
            duration=request.duration,
            temperature=request.temperature,
            guidance_scale=request.guidance_scale,
        )

        # Save as WAV
        sf.write(str(output_path), audio, sr, subtype="PCM_16")
        actual_duration = round(len(audio) / sr, 2)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(exc)}") from exc

    return GenerateResponse(
        output_file_id=output_id,
        prompt=request.prompt,
        duration=actual_duration,
        model=MODEL_ID,
        sample_rate=sr,
    )


@app.get("/output/{file_id}")
async def download_output(file_id: str):
    """Download a generated audio file."""
    output_path = OUTPUT_DIR / f"{file_id}.wav"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail=f"Output '{file_id}' not found.")
    return FileResponse(str(output_path), media_type="audio/wav",
                        filename=f"automix_beat_{file_id}.wav")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
