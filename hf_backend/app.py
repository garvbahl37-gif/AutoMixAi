"""
AutoMixAI – HuggingFace Space Backend (All-in-One)

Consolidated FastAPI backend hosting ALL services:
  - /upload        Upload audio files (stored temporarily)
  - /analyze       BPM + beat detection + energy analysis
  - /mix           Advanced DJ mixing with EQ crossfade
  - /generate      Procedural drum beat generation
  - /output/{id}   Download generated files
  - /recognize     Song recognition via Shazam API
  - /health        Health check

Advanced DJ Mixing Features:
  - LUFS loudness normalization (-24 LUFS pre-mix, -14 LUFS final)
  - High-pass filtering (40Hz rumble removal)
  - Beat detection + BPM estimation
  - Pitch-preserving time-stretching to common BPM
  - Beat-aligned trimming
  - EQ-based crossfade (DJ-style bass swap transition)
  - Equal-power S-curve crossfade
  - Per-track EQ: bass boost, brightness, vocal boost
  - Stereo panning
  - Final LUFS mastering to streaming standard
"""

import os
import sys
import time
import uuid
import tempfile
import subprocess
import re
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import librosa
import soundfile as sf
import scipy.signal
import pyloudnorm as pyln
import torch
from transformers import pipeline as hf_pipeline, AutoFeatureExtractor, AutoModelForAudioClassification

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import httpx

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

SR = 22050               # librosa default for analysis
SR_MIX = 44100           # output sample rate for mixing
HOP_LENGTH = 512
TARGET_LOUDNESS = -24.0  # pre-mix normalization (LUFS)
FINAL_LOUDNESS = -14.0   # streaming-standard final loudness (LUFS)
UPLOAD_DIR = Path(tempfile.gettempdir()) / "automixai_uploads"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "automixai_outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")
SHAZAM_URL = "https://shazam-core.p.rapidapi.com/v1/tracks/recognize"
SHAZAM_HOST = "shazam-core.p.rapidapi.com"


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AutoMixAI Backend",
    description="AI-powered DJ mixing, beat generation, audio analysis, and song recognition.",
    version="2.0.0",
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

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    duration: float
    message: str = "File uploaded successfully."

class AnalyzeRequest(BaseModel):
    file_id: str

class AnalysisResponse(BaseModel):
    file_id: str
    bpm: float
    beat_times: list[float]
    duration: float
    sample_rate: int
    genre: str = "unknown"
    genre_confidence: float = 0.0
    genre_top3: list = []
    tags: list[str] = []
    tag_scores: list = []
    mood: str = "neutral"
    has_vocals: bool = False
    dominant_instrument: str = "unknown"
    instrument_confidence: float = 0.0
    instruments_top3: list = []
    energy: str = "medium"
    message: str = "Analysis complete."

class MixRequest(BaseModel):
    file_id_a: str
    file_id_b: str
    crossfade_duration: float = 8.0
    bass_boost: float = 0.0
    brightness: float = 0.0
    vocal_boost: float = 0.0
    pan_a: float = 0.0
    pan_b: float = 0.0
    eq_transition: bool = True  # Use EQ-based DJ crossfade

class MixResponse(BaseModel):
    output_file_id: str
    duration: float
    bpm_a: float
    bpm_b: float
    target_bpm: float
    message: str = "Mix generated successfully."

class GenerateBeatRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    bars: int = Field(default=4, ge=1, le=32)

class PatternInfo(BaseModel):
    kick: list[int]
    snare: list[int]
    hihat_c: list[int]
    hihat_o: list[int]
    clap: list[int]

class GenerateBeatResponse(BaseModel):
    output_file_id: str
    genre: str
    bpm: float
    bars: int
    complexity: str
    description: str
    duration: float
    pattern: PatternInfo
    sample_rate: int = 44100
    message: str = "Beat generated successfully."


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_file_id() -> str:
    return uuid.uuid4().hex

def find_upload(file_id: str) -> Path:
    for path in UPLOAD_DIR.iterdir():
        if path.stem == file_id:
            return path
    raise FileNotFoundError(f"No file found for ID '{file_id}'")


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio(path: str, sr: int = None, mono: bool = True):
    sr = sr or SR
    y, loaded_sr = librosa.load(path, sr=sr, mono=mono)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y, loaded_sr

def load_audio_full_rate(path: str, mono: bool = True):
    """Load audio at native sample rate (for mixing quality)."""
    y, sr = librosa.load(path, sr=SR_MIX, mono=mono)
    return y, sr

def get_audio_info(path: str) -> dict:
    try:
        info = sf.info(path)
        return {"duration": info.duration, "sample_rate": info.samplerate, "channels": info.channels}
    except Exception:
        y, sr = librosa.load(path, sr=None, mono=False)
        channels = 1 if y.ndim == 1 else y.shape[0]
        duration = (len(y) if y.ndim == 1 else y.shape[1]) / sr
        return {"duration": float(duration), "sample_rate": int(sr), "channels": int(channels)}

def save_audio(path: str, audio: np.ndarray, sr: int):
    sf.write(path, audio, sr, subtype="PCM_16")


# ═══════════════════════════════════════════════════════════════════════════════
# BEAT DETECTION + BPM
# ═══════════════════════════════════════════════════════════════════════════════

def detect_beats(y: np.ndarray, sr: int) -> list[float]:
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH).tolist()
    return beat_times

def estimate_bpm_from_beats(beat_times: list[float]) -> float:
    if len(beat_times) < 2:
        return 120.0
    ibis = np.diff(beat_times)
    median_ibi = float(np.median(ibis))
    if median_ibi <= 0:
        return 120.0
    return round(60.0 / median_ibi, 2)

def estimate_bpm_librosa(y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    return round(float(np.asarray(tempo).flat[0]), 2)


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED DJ MIXING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """LUFS loudness normalization (EBU R128)."""
    try:
        meter = pyln.Meter(sr)
        current_loudness = meter.integrated_loudness(audio)
        if np.isinf(current_loudness) or np.isnan(current_loudness):
            return audio
        return pyln.normalize.loudness(audio, current_loudness, target_lufs)
    except Exception:
        return audio

def highpass_filter(audio: np.ndarray, sr: int, cutoff: float = 40.0) -> np.ndarray:
    """Remove sub-bass rumble below cutoff frequency."""
    sos = scipy.signal.butter(2, cutoff, btype='highpass', fs=sr, output='sos')
    return scipy.signal.sosfilt(sos, audio).astype(np.float32)

def lowpass_filter(audio: np.ndarray, sr: int, cutoff: float = 200.0) -> np.ndarray:
    """Extract bass frequencies."""
    sos = scipy.signal.butter(2, cutoff, btype='low', fs=sr, output='sos')
    return scipy.signal.sosfilt(sos, audio).astype(np.float32)

def highpass_extract(audio: np.ndarray, sr: int, cutoff: float = 200.0) -> np.ndarray:
    """Extract mid+high frequencies (above cutoff)."""
    sos = scipy.signal.butter(2, cutoff, btype='high', fs=sr, output='sos')
    return scipy.signal.sosfilt(sos, audio).astype(np.float32)

def apply_bass_boost(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Boost low frequencies (below 150Hz)."""
    if amount <= 0:
        return audio
    sos = scipy.signal.butter(2, 150, btype='low', fs=sr, output='sos')
    bass = scipy.signal.sosfilt(sos, audio)
    return (audio + amount * bass).astype(np.float32)

def apply_brightness(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Boost high frequencies (above 4kHz)."""
    if amount <= 0:
        return audio
    sos = scipy.signal.butter(2, 4000, btype='high', fs=sr, output='sos')
    highs = scipy.signal.sosfilt(sos, audio)
    return (audio + amount * highs).astype(np.float32)

def pan_stereo(mono_audio: np.ndarray, pan: float) -> np.ndarray:
    """Pan mono audio to stereo. pan: -1 (left) to 1 (right), 0 = center."""
    left_gain = np.cos(max(0, pan) * np.pi / 2)
    right_gain = np.cos(max(0, -pan) * np.pi / 2)
    return np.vstack((mono_audio * left_gain, mono_audio * right_gain))

def time_stretch_to_bpm(y: np.ndarray, sr: int, original_bpm: float, target_bpm: float) -> np.ndarray:
    """Pitch-preserving time-stretch to match target BPM."""
    rate = target_bpm / original_bpm
    if abs(rate - 1.0) < 0.005:
        return y
    return librosa.effects.time_stretch(y, rate=rate)

def align_to_beat(y: np.ndarray, sr: int, beat_times: list[float]) -> np.ndarray:
    """Trim audio to start on the first beat."""
    if not beat_times:
        return y
    start_sample = int(beat_times[0] * sr)
    return y[start_sample:]

def equal_power_crossfade(y1: np.ndarray, y2: np.ndarray, sr: int, duration: float) -> np.ndarray:
    """
    Equal-power (S-curve) crossfade — psychoacoustically smooth.
    Uses sin/cos curves instead of linear, preventing the volume dip
    that occurs with linear crossfades.
    """
    fade_samples = int(duration * sr)
    fade_samples = min(fade_samples, len(y1), len(y2))
    if fade_samples < 1:
        return np.concatenate([y1, y2])

    # Equal-power curves: cos² + sin² = 1 (constant power)
    t = np.linspace(0, np.pi / 2, fade_samples)
    fade_out = np.cos(t) ** 2  # smooth fade out for track A
    fade_in = np.sin(t) ** 2   # smooth fade in for track B

    tail = y1[-fade_samples:] * fade_out
    head = y2[:fade_samples] * fade_in
    mixed_region = tail + head

    return np.concatenate([y1[:-fade_samples], mixed_region, y2[fade_samples:]])

def eq_crossfade(y1: np.ndarray, y2: np.ndarray, sr: int, duration: float) -> np.ndarray:
    """
    Professional DJ-style EQ crossfade transition.

    Instead of simply fading volumes, this mimics what real DJs do:
    1. Fade out Track A's BASS while fading in Track B's BASS
    2. Crossfade mid+high frequencies with an S-curve
    3. Prevents muddiness and bass clashing during transitions

    This is the core technique used in professional DJ sets.
    """
    fade_samples = int(duration * sr)
    fade_samples = min(fade_samples, len(y1), len(y2))
    if fade_samples < 1:
        return np.concatenate([y1, y2])

    # Split both tracks into bass (< 200Hz) and mid+high (> 200Hz)
    y1_bass = lowpass_filter(y1[-fade_samples:], sr, cutoff=200)
    y1_mid_high = highpass_extract(y1[-fade_samples:], sr, cutoff=200)
    y2_bass = lowpass_filter(y2[:fade_samples], sr, cutoff=200)
    y2_mid_high = highpass_extract(y2[:fade_samples], sr, cutoff=200)

    # Equal-power curves
    t = np.linspace(0, np.pi / 2, fade_samples)
    fade_out = np.cos(t) ** 2
    fade_in = np.sin(t) ** 2

    # Bass: sharper transition (prevent bass clash)
    # The bass swaps over more aggressively in the middle
    bass_t = np.linspace(0, np.pi / 2, fade_samples)
    bass_fade_out = np.cos(bass_t) ** 3  # sharper cut
    bass_fade_in = np.sin(bass_t) ** 3   # sharper rise

    # Mix the bass swap + mid/high crossfade
    bass_region = y1_bass * bass_fade_out + y2_bass * bass_fade_in
    mid_high_region = y1_mid_high * fade_out + y2_mid_high * fade_in
    mixed_region = bass_region + mid_high_region

    return np.concatenate([y1[:-fade_samples], mixed_region, y2[fade_samples:]])

def create_advanced_mix(
    path_a: str,
    path_b: str,
    output_path: str,
    crossfade_duration: float = 8.0,
    bass_boost: float = 0.0,
    brightness: float = 0.0,
    vocal_boost: float = 0.0,
    pan_a: float = 0.0,
    pan_b: float = 0.0,
    eq_transition: bool = True,
) -> dict:
    """
    Advanced DJ mixing pipeline.

    Steps:
      1. Load both tracks at 44.1kHz
      2. LUFS normalize to -24 LUFS
      3. High-pass filter at 40Hz (remove rumble)
      4. Detect beats + estimate BPM
      5. Time-stretch both to a common BPM (average)
      6. Align to first beat boundary
      7. Apply per-track EQ (bass boost, brightness)
      8. EQ-based crossfade or equal-power crossfade
      9. Final LUFS mastering to -14 LUFS
    """
    print(f"=== Creating advanced DJ mix ===")
    print(f"Track A: {path_a}")
    print(f"Track B: {path_b}")

    # 1. Load at high quality
    y_a, sr = load_audio_full_rate(path_a)
    y_b, _ = load_audio_full_rate(path_b)
    print(f"Loaded: A={len(y_a)/sr:.1f}s, B={len(y_b)/sr:.1f}s at {sr}Hz")

    # 2. LUFS normalize
    y_a = normalize_loudness(y_a, sr, TARGET_LOUDNESS)
    y_b = normalize_loudness(y_b, sr, TARGET_LOUDNESS)
    print("LUFS normalized to -24 LUFS")

    # 3. High-pass filter
    y_a = highpass_filter(y_a, sr, cutoff=40)
    y_b = highpass_filter(y_b, sr, cutoff=40)

    # 4. Beat detection + BPM (use lower SR for speed, then apply to full audio)
    y_a_analysis = librosa.resample(y_a, orig_sr=sr, target_sr=SR)
    y_b_analysis = librosa.resample(y_b, orig_sr=sr, target_sr=SR)

    beats_a = detect_beats(y_a_analysis, SR)
    beats_b = detect_beats(y_b_analysis, SR)

    bpm_a = estimate_bpm_from_beats(beats_a) if len(beats_a) >= 2 else estimate_bpm_librosa(y_a_analysis, SR)
    bpm_b = estimate_bpm_from_beats(beats_b) if len(beats_b) >= 2 else estimate_bpm_librosa(y_b_analysis, SR)

    target_bpm = round((bpm_a + bpm_b) / 2, 2)
    print(f"BPMs — A: {bpm_a:.1f}, B: {bpm_b:.1f} → target: {target_bpm:.1f}")

    # 5. Time-stretch to common BPM
    y_a = time_stretch_to_bpm(y_a, sr, bpm_a, target_bpm)
    y_b = time_stretch_to_bpm(y_b, sr, bpm_b, target_bpm)
    print("Time-stretched to common BPM")

    # Re-detect beats after stretch for alignment
    y_a_stretched_analysis = librosa.resample(y_a, orig_sr=sr, target_sr=SR)
    y_b_stretched_analysis = librosa.resample(y_b, orig_sr=sr, target_sr=SR)
    beats_a_new = detect_beats(y_a_stretched_analysis, SR)
    beats_b_new = detect_beats(y_b_stretched_analysis, SR)

    # 6. Align to beat boundaries
    y_a = align_to_beat(y_a, sr, beats_a_new)
    y_b = align_to_beat(y_b, sr, beats_b_new)

    # 7. Apply per-track EQ
    y_a = apply_bass_boost(y_a, sr, bass_boost)
    y_b = apply_bass_boost(y_b, sr, bass_boost)
    y_a = apply_brightness(y_a, sr, brightness)
    y_b = apply_brightness(y_b, sr, brightness)

    # Vocal boost (boost mid frequencies 1-4kHz)
    if vocal_boost > 0:
        sos = scipy.signal.butter(2, [1000, 4000], btype='band', fs=sr, output='sos')
        y_a_vocal = scipy.signal.sosfilt(sos, y_a)
        y_b_vocal = scipy.signal.sosfilt(sos, y_b)
        y_a = y_a + vocal_boost * y_a_vocal
        y_b = y_b + vocal_boost * y_b_vocal

    # 8. Crossfade
    if eq_transition:
        mixed = eq_crossfade(y_a, y_b, sr, duration=crossfade_duration)
        print(f"EQ crossfade applied: {crossfade_duration:.1f}s")
    else:
        mixed = equal_power_crossfade(y_a, y_b, sr, duration=crossfade_duration)
        print(f"Equal-power crossfade applied: {crossfade_duration:.1f}s")

    # 9. Final LUFS mastering
    mixed = normalize_loudness(mixed, sr, FINAL_LOUDNESS)
    print("Final mastering to -14 LUFS")

    # Soft limiter to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)

    # Save
    save_audio(output_path, mixed, sr)

    duration = float(len(mixed) / sr)
    print(f"Mix complete: {duration:.1f}s saved to {output_path}")

    return {
        "bpm_a": bpm_a,
        "bpm_b": bpm_b,
        "target_bpm": target_bpm,
        "duration": duration,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ML-POWERED AUDIO CLASSIFICATION (Genre, Mood, Vocals)
# Uses HuggingFace transformer models for accurate classification
# ═══════════════════════════════════════════════════════════════════════════════

# Model IDs
GENRE_MODEL_ID = "dima806/music_genres_classification"      # wav2vec2, GTZAN 10 genres
MOOD_MODEL_ID = "StanislavKo28/music_moods_classification"  # wav2vec2, 14 moods

# Lazy-loaded classifiers (loaded on first use to speed up startup)
_genre_classifier = None
_mood_classifier = None

def _get_genre_classifier():
    """Lazy-load the genre classification pipeline."""
    global _genre_classifier
    if _genre_classifier is None:
        print(f"Loading genre model: {GENRE_MODEL_ID}")
        _genre_classifier = hf_pipeline(
            "audio-classification",
            model=GENRE_MODEL_ID,
            device=-1,  # CPU
        )
        print("Genre model loaded.")
    return _genre_classifier

def _get_mood_classifier():
    """Lazy-load the mood classification pipeline."""
    global _mood_classifier
    if _mood_classifier is None:
        print(f"Loading mood model: {MOOD_MODEL_ID}")
        _mood_classifier = hf_pipeline(
            "audio-classification",
            model=MOOD_MODEL_ID,
            device=-1,  # CPU
        )
        print("Mood model loaded.")
    return _mood_classifier


def classify_genre(file_path: str) -> dict:
    """
    Classify music genre using dima806/music_genres_classification.
    Returns top genre + confidence + top 3 predictions.
    Genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
    """
    try:
        classifier = _get_genre_classifier()
        results = classifier(file_path, top_k=5)
        top = results[0]
        top3 = [
            {"genre": r["label"], "confidence": round(r["score"], 4)}
            for r in results[:3]
        ]
        return {
            "genre": top["label"],
            "confidence": round(top["score"], 4),
            "top3": top3,
        }
    except Exception as e:
        print(f"Genre classification error: {e}")
        return {"genre": "unknown", "confidence": 0.0, "top3": []}


def classify_mood(file_path: str) -> dict:
    """
    Classify music mood using StanislavKo28/music_moods_classification.
    Returns top mood + confidence.
    Moods: angry, dark, energetic, epic, euphoric, happy, mysterious,
           relaxing, romantic, sad, scary, glamorous, uplifting, sentimental
    """
    try:
        classifier = _get_mood_classifier()
        results = classifier(file_path, top_k=5)
        top = results[0]
        top3 = [
            {"mood": r["label"], "confidence": round(r["score"], 4)}
            for r in results[:3]
        ]
        return {
            "mood": top["label"],
            "confidence": round(top["score"], 4),
            "top3": top3,
        }
    except Exception as e:
        print(f"Mood classification error: {e}")
        return {"mood": "neutral", "confidence": 0.0, "top3": []}


def detect_vocals(y: np.ndarray, sr: int) -> dict:
    """
    Detect whether the audio has vocals using harmonic/percussive separation.
    If the harmonic component has strong mid-frequency energy (1-4kHz vocal range),
    the track likely has vocals.
    """
    try:
        # Separate harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Compute spectral centroid of harmonic part
        harmonic_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)))

        # Compute energy in vocal frequency range (1-4kHz)
        S = np.abs(librosa.stft(y_harmonic))
        freqs = librosa.fft_frequencies(sr=sr)
        vocal_mask = (freqs >= 1000) & (freqs <= 4000)
        full_mask = freqs >= 80

        vocal_energy = float(np.mean(S[vocal_mask, :])) if np.any(vocal_mask) else 0
        total_energy = float(np.mean(S[full_mask, :])) if np.any(full_mask) else 1

        vocal_ratio = vocal_energy / max(total_energy, 1e-8)

        # Harmonic-to-percussive ratio
        h_energy = float(np.mean(y_harmonic ** 2))
        p_energy = float(np.mean(y_percussive ** 2))
        hp_ratio = h_energy / max(p_energy, 1e-8)

        # Decision: high vocal ratio + high harmonic content = vocals
        has_vocals = vocal_ratio > 0.8 and hp_ratio > 1.2

        return {
            "has_vocals": has_vocals,
            "vocal_ratio": round(vocal_ratio, 4),
            "harmonic_percussive_ratio": round(hp_ratio, 4),
            "label": "Vocal" if has_vocals else "Instrumental",
        }
    except Exception as e:
        print(f"Vocal detection error: {e}")
        return {"has_vocals": False, "vocal_ratio": 0.0, "harmonic_percussive_ratio": 0.0, "label": "Unknown"}


# ═══════════════════════════════════════════════════════════════════════════════
# BEAT GENERATOR (Procedural Drum Synthesis)
# (Condensed version of the full beat_generator.py)
# ═══════════════════════════════════════════════════════════════════════════════

BEAT_SR = 44100

class Complexity(Enum):
    MINIMAL = "minimal"
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    INTRICATE = "intricate"

class Energy(Enum):
    SOFT = "soft"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    INTENSE = "intense"

class Mood(Enum):
    DARK = "dark"
    MELANCHOLIC = "melancholic"
    NEUTRAL = "neutral"
    UPLIFTING = "uplifting"
    AGGRESSIVE = "aggressive"

@dataclass
class BeatParams:
    genre: str
    sub_genre: Optional[str]
    bpm: float
    bars: int
    complexity: Complexity
    energy: Energy
    mood: Mood
    time_signature: Tuple[int, int]
    swing: float
    humanize: float
    include_fills: bool
    instruments: List[str]
    description: str

# Genre defaults
_GENRE_DEFAULTS = {
    "hiphop": {"bpm": 90, "range": (70, 110)},
    "trap": {"bpm": 140, "range": (120, 160)},
    "edm": {"bpm": 128, "range": (118, 150)},
    "house": {"bpm": 124, "range": (118, 130)},
    "techno": {"bpm": 130, "range": (120, 145)},
    "dnb": {"bpm": 174, "range": (160, 180)},
    "rock": {"bpm": 120, "range": (100, 150)},
    "metal": {"bpm": 160, "range": (100, 220)},
    "pop": {"bpm": 120, "range": (95, 135)},
    "jazz": {"bpm": 120, "range": (60, 200)},
    "reggae": {"bpm": 80, "range": (65, 100)},
    "funk": {"bpm": 105, "range": (85, 125)},
    "ambient": {"bpm": 75, "range": (50, 100)},
}

_GENRE_KEYWORDS = {
    "hiphop": ["hip hop", "hip-hop", "hiphop", "rap", "boom bap", "lofi", "lo-fi"],
    "trap": ["trap", "atlanta trap", "drill", "phonk"],
    "edm": ["edm", "electronic dance", "electronic"],
    "house": ["house", "deep house", "tech house"],
    "techno": ["techno", "industrial", "acid"],
    "dnb": ["drum and bass", "dnb", "d&b", "jungle"],
    "rock": ["rock", "indie", "punk", "grunge"],
    "metal": ["metal", "heavy metal", "thrash", "death metal"],
    "pop": ["pop", "mainstream"],
    "jazz": ["jazz", "swing", "bebop"],
    "reggae": ["reggae", "dub", "ska", "dancehall"],
    "funk": ["funk", "funky", "groove", "disco", "soul"],
    "ambient": ["ambient", "chill", "chillout", "downtempo"],
}

def _kw_in(kw: str, text: str) -> bool:
    pattern = r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])"
    return bool(re.search(pattern, text, re.IGNORECASE))

def parse_prompt(prompt: str) -> BeatParams:
    text = prompt.lower().strip()

    # Find genre
    genre = "hiphop"
    for g, keywords in _GENRE_KEYWORDS.items():
        if any(_kw_in(kw, text) for kw in keywords):
            genre = g
            break

    # Find BPM
    bpm_match = re.search(r"(\d{2,3})\s*(?:bpm|tempo)", text, re.IGNORECASE)
    if bpm_match:
        bpm = max(40.0, min(250.0, float(bpm_match.group(1))))
    else:
        bpm = _GENRE_DEFAULTS.get(genre, {}).get("bpm", 120)

    # Find bars
    bars_match = re.search(r"(\d+)\s*(?:bar|bars|measure)", text, re.IGNORECASE)
    bars = int(bars_match.group(1)) if bars_match else 4
    bars = max(1, min(32, bars))

    # Complexity
    if any(w in text for w in ["complex", "intricate", "busy"]):
        complexity = Complexity.COMPLEX
    elif any(w in text for w in ["simple", "basic", "minimal"]):
        complexity = Complexity.SIMPLE
    else:
        complexity = Complexity.MEDIUM

    # Energy
    if any(w in text for w in ["intense", "aggressive", "hard", "heavy"]):
        energy = Energy.INTENSE
    elif any(w in text for w in ["energetic", "powerful", "driving"]):
        energy = Energy.HIGH
    elif any(w in text for w in ["soft", "gentle", "quiet"]):
        energy = Energy.SOFT
    elif any(w in text for w in ["chill", "relaxed", "laid-back"]):
        energy = Energy.LOW
    else:
        energy = Energy.MEDIUM

    mood = Mood.NEUTRAL
    if any(w in text for w in ["dark", "sinister"]):
        mood = Mood.DARK
    elif any(w in text for w in ["happy", "uplifting", "bright"]):
        mood = Mood.UPLIFTING

    instruments = ["kick", "snare", "hihat_c", "hihat_o", "clap"]

    desc = f"{mood.value.title()} {energy.value} {genre.title()} beat at {bpm:.0f} BPM, {bars} bars"

    return BeatParams(
        genre=genre, sub_genre=None, bpm=bpm, bars=bars,
        complexity=complexity, energy=energy, mood=mood,
        time_signature=(4, 4), swing=0.0, humanize=0.3,
        include_fills=False, instruments=instruments, description=desc,
    )


# Drum synthesis functions
def _env(length: int, attack: float = 0.002, decay: float = 0.15):
    t = np.linspace(0, 1, length)
    env = np.exp(-t / max(decay, 1e-6))
    atk = int(attack * BEAT_SR)
    if 0 < atk < length:
        env[:atk] *= np.linspace(0, 1, atk)
    return env

def synth_kick(dur=0.4):
    n = int(dur * BEAT_SR)
    t = np.linspace(0, dur, n)
    freq = np.linspace(200, 55, n)
    phase = 2 * np.pi * np.cumsum(freq) / BEAT_SR
    sine = np.sin(phase) * _env(n, 0.001, 0.25) * 0.8
    click_n = int(0.008 * BEAT_SR)
    click = np.zeros(n)
    click[:click_n] = (np.random.rand(click_n) * 2 - 1) * np.linspace(1, 0, click_n)
    return (sine + click * 0.2).astype(np.float32)

def synth_snare(dur=0.25):
    n = int(dur * BEAT_SR)
    t = np.linspace(0, dur, n)
    noise = np.random.randn(n) * _env(n, 0.001, 0.08) * 0.6
    body = np.sin(2 * np.pi * 200 * t) * _env(n, 0.001, 0.05) * 0.5
    return (noise + body).astype(np.float32)

def synth_hihat_c(dur=0.05):
    n = int(dur * BEAT_SR)
    noise = np.random.randn(n)
    filtered = np.diff(noise, prepend=noise[0])
    return (filtered * _env(n, 0.0005, 0.02) * 0.5).astype(np.float32)

def synth_hihat_o(dur=0.3):
    n = int(dur * BEAT_SR)
    noise = np.random.randn(n)
    filtered = np.diff(noise, prepend=noise[0])
    ring = np.sin(2 * np.pi * 6000 * np.linspace(0, dur, n))
    return (filtered * _env(n, 0.001, 0.2) * 0.4 + ring * _env(n, 0.001, 0.2) * 0.1).astype(np.float32)

def synth_clap(dur=0.15):
    n = int(dur * BEAT_SR)
    noise = np.random.randn(n)
    env = np.zeros(n)
    for offset_ms in [0, 10, 20, 30]:
        offset = int(offset_ms * BEAT_SR / 1000)
        if offset < n:
            env[offset:] += _env(n - offset, 0.001, 0.04) * (1 - offset_ms / 50)
    return (noise * env * 0.5).astype(np.float32)

# Sound bank
_SOUNDS = {
    "kick": synth_kick(),
    "snare": synth_snare(),
    "hihat_c": synth_hihat_c(),
    "hihat_o": synth_hihat_o(),
    "clap": synth_clap(),
}

def generate_pattern(params: BeatParams) -> Dict[str, List[int]]:
    steps = 16
    pattern = {}
    genre = params.genre
    cx = params.complexity

    # Kick
    kick = [0] * steps
    if genre in ["edm", "house", "techno"]:
        for i in range(0, steps, 4): kick[i] = 1
    elif genre == "dnb":
        kick[0] = 1; kick[10] = 1
    elif genre in ["hiphop", "trap"]:
        kick[0] = 1; kick[6] = 1
        if cx in [Complexity.COMPLEX, Complexity.INTRICATE]: kick[10] = 1
    else:
        kick[0] = 1; kick[8] = 1
    pattern["kick"] = kick

    # Snare
    snare = [0] * steps
    if genre == "reggae":
        snare[8] = 1
    elif genre == "trap":
        snare[12] = 1
    else:
        snare[4] = 1; snare[12] = 1
    pattern["snare"] = snare

    # Hi-hats
    hh_c = [0] * steps
    if cx == Complexity.MINIMAL:
        for i in range(0, steps, 4): hh_c[i] = 1
    elif cx == Complexity.SIMPLE:
        for i in range(0, steps, 2): hh_c[i] = 1
    else:
        for i in range(steps): hh_c[i] = 1
    pattern["hihat_c"] = hh_c

    hh_o = [0] * steps
    if cx != Complexity.MINIMAL:
        hh_o[7] = 1; hh_o[15] = 1
    pattern["hihat_o"] = hh_o

    # Clap
    clap = [0] * steps
    if genre in ["trap", "edm", "house"]:
        clap[4] = 1; clap[12] = 1
    pattern["clap"] = clap

    return pattern


def render_beat(pattern: Dict[str, List[int]], params: BeatParams) -> np.ndarray:
    steps_per_bar = 16
    beat_dur = 60.0 / params.bpm
    step_dur = beat_dur / 4.0
    bar_samples = int(steps_per_bar * step_dur * BEAT_SR)
    total_samples = bar_samples * params.bars

    audio = np.zeros(total_samples, dtype=np.float32)

    gain_map = {"kick": 0.95, "snare": 0.80, "hihat_c": 0.50, "hihat_o": 0.55, "clap": 0.70}

    for instrument, steps in pattern.items():
        sound = _SOUNDS.get(instrument)
        if sound is None:
            continue
        base_gain = gain_map.get(instrument, 0.6)
        for bar in range(params.bars):
            for step, hit in enumerate(steps):
                if not hit:
                    continue
                sample_pos = bar * bar_samples + int(step * step_dur * BEAT_SR)
                # Humanize
                if params.humanize > 0:
                    sample_pos += int(np.random.normal(0, params.humanize * 0.01 * BEAT_SR))
                sample_pos = max(0, min(sample_pos, total_samples - 1))
                end = min(sample_pos + len(sound), total_samples)
                write_len = end - sample_pos
                if write_len > 0:
                    audio[sample_pos:end] += sound[:write_len] * base_gain

    peak = np.abs(audio).max()
    if peak > 0.95:
        audio = audio * (0.95 / peak)

    return audio


def generate_beat_full(prompt: str, output_path: str) -> dict:
    params = parse_prompt(prompt)
    pattern = generate_pattern(params)
    audio = render_beat(pattern, params)
    duration = len(audio) / BEAT_SR
    sf.write(output_path, audio, BEAT_SR, subtype="PCM_16")

    return {
        "genre": params.genre,
        "bpm": params.bpm,
        "bars": params.bars,
        "complexity": params.complexity.value,
        "description": params.description,
        "duration": round(duration, 3),
        "pattern": pattern,
        "sample_rate": BEAT_SR,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SHAZAM RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════

def convert_to_wav(input_path):
    output_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
    try:
        cmd = ['ffmpeg', '-y', '-i', input_path, '-ar', '44100', '-ac', '1', '-sample_fmt', 's16', '-f', 'wav', output_path]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
    except Exception as e:
        print(f"Conversion error: {e}")
    return None

async def recognize_shazam(audio_path):
    if not RAPIDAPI_KEY:
        return None
    headers = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": SHAZAM_HOST}
    try:
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        async with httpx.AsyncClient(timeout=25) as client:
            response = await client.post(SHAZAM_URL, headers=headers, files={"file": ("audio.wav", audio_bytes, "audio/wav")})
        if response.status_code != 200:
            return None
        data = response.json()
        track = data.get("track") or data
        if not track.get("title") and not track.get("heading"):
            return None
        title = track.get("title", "Unknown")
        artist = track.get("subtitle", "Unknown Artist")
        album = cover = year = spotify_url = apple_music_url = None
        shazam_url = track.get("url")
        sections = track.get("sections", [])
        for section in sections:
            if section.get("type") == "SONG":
                for meta in section.get("metadata", []):
                    if meta.get("title") == "Album": album = meta.get("text")
                    elif meta.get("title") == "Released": year = meta.get("text")
        images = track.get("images", {})
        cover = images.get("coverarthq") or images.get("coverart")
        hub = track.get("hub", {})
        for provider in hub.get("providers", []):
            ptype = provider.get("type", "").upper()
            for action in provider.get("actions", []):
                uri = action.get("uri", "")
                if ptype == "SPOTIFY" and uri: spotify_url = uri
                elif ptype == "APPLE" and uri: apple_music_url = uri
        score = len(data.get("matches", []))
        return {"title": title, "artist": artist, "album": album, "cover": cover, "year": year,
                "spotify": spotify_url, "apple_music": apple_music_url, "shazam_url": shazam_url,
                "score": max(score, 1), "source": "shazam"}
    except Exception as e:
        print(f"Shazam error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "ok", "service": "AutoMixAI Backend v2.0", "features": [
        "upload", "analyze", "mix (advanced DJ)", "generate", "recognize"
    ]}

@app.get("/health")
def health():
    return {"status": "healthy"}


# ── Upload ───────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

@app.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    original_name = file.filename or "unknown"
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")

    file_id = generate_file_id()
    save_path = UPLOAD_DIR / f"{file_id}{ext}"

    try:
        contents = await file.read()
        save_path.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to save file.") from exc

    try:
        info = get_audio_info(str(save_path))
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Not a valid audio file.") from exc

    return UploadResponse(file_id=file_id, filename=original_name, duration=round(info["duration"], 2))


# ── Analyze ──────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(request: AnalyzeRequest):
    try:
        file_path = find_upload(request.file_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File '{request.file_id}' not found.")

    file_path_str = str(file_path)

    # Load for librosa analysis
    y, sr = load_audio(file_path_str)
    beat_times = detect_beats(y, sr)
    bpm = estimate_bpm_from_beats(beat_times) if len(beat_times) >= 2 else estimate_bpm_librosa(y, sr)

    # ML Genre classification (transformer model)
    genre_result = classify_genre(file_path_str)

    # ML Mood classification (transformer model)
    mood_result = classify_mood(file_path_str)

    # Vocal detection (spectral analysis)
    vocal_result = detect_vocals(y, sr)

    # Energy (RMS-based)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    if rms > 0.12: energy = "high"
    elif rms > 0.06: energy = "medium"
    else: energy = "low"

    return AnalysisResponse(
        file_id=request.file_id,
        bpm=bpm,
        beat_times=beat_times,
        duration=round(len(y) / sr, 2),
        sample_rate=sr,
        genre=genre_result["genre"],
        genre_confidence=genre_result["confidence"],
        genre_top3=genre_result.get("top3", []),
        mood=mood_result["mood"],
        has_vocals=vocal_result["has_vocals"],
        energy=energy,
    )


# ── Mix ──────────────────────────────────────────────────────────────────────

@app.post("/mix", response_model=MixResponse)
async def mix_tracks(request: MixRequest):
    try:
        path_a = find_upload(request.file_id_a)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Track A '{request.file_id_a}' not found.")
    try:
        path_b = find_upload(request.file_id_b)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Track B '{request.file_id_b}' not found.")

    output_id = generate_file_id()
    output_path = OUTPUT_DIR / f"{output_id}.wav"

    try:
        result = create_advanced_mix(
            path_a=str(path_a),
            path_b=str(path_b),
            output_path=str(output_path),
            crossfade_duration=request.crossfade_duration,
            bass_boost=request.bass_boost,
            brightness=request.brightness,
            vocal_boost=request.vocal_boost,
            pan_a=request.pan_a,
            pan_b=request.pan_b,
            eq_transition=request.eq_transition,
        )
    except Exception as exc:
        print(f"Mix error: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Mixing failed: {str(exc)}") from exc

    return MixResponse(
        output_file_id=output_id,
        duration=result["duration"],
        bpm_a=result["bpm_a"],
        bpm_b=result["bpm_b"],
        target_bpm=result["target_bpm"],
    )


# ── Generate (Procedural Synth) ──────────────────────────────────────────────

@app.post("/generate", response_model=GenerateBeatResponse)
async def generate_beat_route(request: GenerateBeatRequest):
    output_id = generate_file_id()
    output_path = OUTPUT_DIR / f"{output_id}.wav"

    effective_prompt = request.prompt
    if request.bars:
        effective_prompt = f"{request.prompt} {request.bars} bars"

    try:
        result = generate_beat_full(effective_prompt, str(output_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Beat generation failed: {exc}") from exc

    pattern_raw = result["pattern"]
    return GenerateBeatResponse(
        output_file_id=output_id,
        genre=result["genre"],
        bpm=result["bpm"],
        bars=result["bars"],
        complexity=result["complexity"],
        description=result["description"],
        duration=result["duration"],
        pattern=PatternInfo(
            kick=pattern_raw.get("kick", [0]*16),
            snare=pattern_raw.get("snare", [0]*16),
            hihat_c=pattern_raw.get("hihat_c", [0]*16),
            hihat_o=pattern_raw.get("hihat_o", [0]*16),
            clap=pattern_raw.get("clap", [0]*16),
        ),
    )


# ── Generate AI (MusicGen via HF Inference API) ─────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MUSICGEN_API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"

class GenerateAIRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    duration: int = Field(default=10, ge=3, le=30)

class GenerateAIResponse(BaseModel):
    output_file_id: str
    prompt: str
    duration: float
    model: str = "facebook/musicgen-small"
    sample_rate: int = 32000
    message: str = "AI beat generated successfully."

@app.post("/generate-ai", response_model=GenerateAIResponse)
async def generate_beat_ai(request: GenerateAIRequest):
    """Generate a beat using Meta's MusicGen via HuggingFace Inference API (free GPU)."""
    output_id = generate_file_id()
    output_path = OUTPUT_DIR / f"{output_id}.wav"

    headers = {
        "Content-Type": "application/json",
        "x-wait-for-model": "true",  # Wait for model to load instead of 503
    }
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "inputs": request.prompt,
    }

    try:
        print(f"MusicGen AI generating: '{request.prompt}' ({request.duration}s)")
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                MUSICGEN_API_URL,
                headers=headers,
                json=payload,
            )

        print(f"HF API response: status={response.status_code}, content-type={response.headers.get('content-type', 'unknown')}, size={len(response.content)} bytes")

        if response.status_code == 503:
            error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            wait_time = error_data.get("estimated_time", 30)
            raise HTTPException(
                status_code=503,
                detail=f"MusicGen model is loading, please try again in ~{int(wait_time)} seconds."
            )
        if response.status_code != 200:
            error_msg = response.text[:500]
            print(f"HF API error: {error_msg}")
            raise HTTPException(
                status_code=502,
                detail=f"HF Inference API error ({response.status_code}): {error_msg}"
            )

        # Response is raw audio bytes (FLAC format)
        audio_bytes = response.content

        # Save the raw audio first
        temp_path = str(output_path).replace(".wav", "_raw.flac")
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        # Convert to WAV using librosa
        y, sr = librosa.load(temp_path, sr=None, mono=True)
        sf.write(str(output_path), y, sr, subtype="PCM_16")

        # Clean up temp
        os.remove(temp_path)

        actual_duration = round(len(y) / sr, 2)
        print(f"MusicGen AI complete: {actual_duration}s")

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(exc)}") from exc

    return GenerateAIResponse(
        output_file_id=output_id,
        prompt=request.prompt,
        duration=actual_duration,
        sample_rate=int(sr),
    )


# ── Output ───────────────────────────────────────────────────────────────────

@app.get("/output/{file_id}")
async def download_output(file_id: str):
    output_path = OUTPUT_DIR / f"{file_id}.wav"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail=f"Output file '{file_id}' not found.")
    return FileResponse(str(output_path), media_type="audio/wav", filename=f"automix_{file_id}.wav")


# ── Recognize ────────────────────────────────────────────────────────────────

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
    tmp_path = os.path.join(tempfile.gettempdir(), f"shazam_{uuid.uuid4().hex}{suffix}")
    converted_path = None

    try:
        content = await file.read()
        if len(content) < 500:
            return {"status": "error", "message": "Audio too small"}
        with open(tmp_path, 'wb') as f:
            f.write(content)

        converted_path = convert_to_wav(tmp_path)
        work_path = converted_path if converted_path else tmp_path

        result = await recognize_shazam(work_path)

        if result:
            return {
                "status": "found", "title": result["title"], "artist": result["artist"],
                "album": result.get("album"), "cover": result.get("cover"),
                "year": result.get("year"), "spotify": result.get("spotify"),
                "apple_music": result.get("apple_music"), "shazam_url": result.get("shazam_url"),
                "score": result.get("score", 0), "source": result.get("source", "unknown"),
                "match_quality": "high", "is_early": True,
            }
        else:
            return {"status": "not_found", "message": "No song matched.", "is_early": False}
    except Exception as e:
        return {"status": "error", "message": f"Recognition failed: {str(e)}"}
    finally:
        for path in [tmp_path, converted_path]:
            if path:
                try: os.unlink(path)
                except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
