"""
AutoMixAI – Beat Generator Service

Generates a synthesised drum beat from a natural-language prompt.

Pipeline:
  1. parse_prompt()   — extract BPM, genre / style, bars, complexity
  2. get_pattern()    — 16-step grid patterns per instrument
  3. synthesize_*()   — pure-numpy drum sound synthesis
  4. render_beat()    — stitch everything into a waveform
  5. generate_beat()  — public entry-point, saves WAV and returns metadata
"""

from __future__ import annotations

import re
import math
import numpy as np
from pathlib import Path
from typing import Optional

import soundfile as sf

from app.utils.logger import get_logger

logger = get_logger(__name__)

SR = 44100   # internal sample rate for generation


# ══════════════════════════════════════════════════════════════════════
# 1.  PROMPT PARSER
# ══════════════════════════════════════════════════════════════════════

_GENRE_KEYWORDS: dict[str, list[str]] = {
    "hiphop":    ["hip hop", "hip-hop", "hiphop", "rap", "boom bap", "boom-bap",
                  "lo-fi", "lofi", "lo fi"],
    "trap":      ["trap", "dark trap", "plugg", "atlanta"],
    "edm":       ["edm", "house", "electronic", "dance", "techno", "trance",
                  "festival", "club", "4x4"],
    "rock":      ["rock", "punk", "grunge", "garage", "indie rock"],
    "metal":     ["metal", "heavy metal", "thrash", "death metal", "hardcore"],
    "jazz":      ["jazz", "swing", "bebop", "jazz fusion", "jazzy"],
    "reggae":    ["reggae", "reggaeton", "dancehall", "one drop", "ska"],
    "dnb":       ["drum and bass", "drum & bass", "dnb", "jungle", "liquid dnb",
                  "neurofunk"],
    "ambient":   ["ambient", "atmospheric", "chill", "chillout", "meditation",
                  "relaxing", "slow"],
    "afrobeats": ["afrobeats", "afro", "amapiano", "afropop"],
    "funk":      ["funk", "funky", "groove", "soul"],
    "latin":     ["latin", "salsa", "bossa nova", "cumbia"],
}

_BPM_DEFAULTS: dict[str, float] = {
    "hiphop": 90, "trap": 140, "edm": 128, "rock": 130, "metal": 160,
    "jazz": 120, "reggae": 80, "dnb": 174, "ambient": 75,
    "afrobeats": 112, "funk": 105, "latin": 100,
}

_COMPLEXITY_KEYWORDS = {
    "complex":  ["complex", "busy", "filled", "intricate", "polyrhythm"],
    "simple":   ["simple", "minimal", "sparse", "basic", "clean", "stripped"],
}


def _kw_in(kw: str, text: str) -> bool:
    """
    Case-insensitive keyword match that respects word boundaries.
    Prevents "rap" (hiphop) from being found inside "trap", etc.
    """
    return bool(re.search(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])", text))


def parse_prompt(prompt: str) -> dict:
    """
    Extract structured parameters from a free-text prompt.

    Returns::
        {
          "genre": "hiphop",
          "bpm": 90.0,
          "bars": 4,
          "complexity": "medium",    # "simple" | "medium" | "complex"
          "description": "chill hip-hop beat at 90 BPM"
        }
    """
    text = prompt.lower().strip()

    # ── detect genre (longest keyword first to avoid partial matches) ─
    matched_genre = "hiphop"   # default
    for genre, kws in _GENRE_KEYWORDS.items():
        # Sort by length descending so "drum and bass" wins over "bass"
        if any(_kw_in(kw, text) for kw in sorted(kws, key=len, reverse=True)):
            matched_genre = genre
            break

    # ── detect explicit BPM ───────────────────────────────────────────
    bpm_match = re.search(r"(\d{2,3})\s*(?:bpm|tempo|beats)", text)
    if bpm_match:
        bpm = float(bpm_match.group(1))
        bpm = max(40.0, min(220.0, bpm))
    else:
        # modifier words
        if any(w in text for w in ["fast", "upbeat", "energetic", "hard"]):
            bpm = _BPM_DEFAULTS.get(matched_genre, 120) * 1.15
        elif any(w in text for w in ["slow", "relaxed", "chill", "laid-back"]):
            bpm = _BPM_DEFAULTS.get(matched_genre, 120) * 0.85
        else:
            bpm = _BPM_DEFAULTS.get(matched_genre, 120)
        bpm = round(bpm, 1)

    # ── detect bar count ──────────────────────────────────────────────
    bar_match = re.search(r"(\d+)\s*bar", text)
    if bar_match:
        bars = int(bar_match.group(1))
        bars = max(1, min(32, bars))
    else:
        bars = 4

    # ── complexity ────────────────────────────────────────────────────
    complexity = "medium"
    for level, kws in _COMPLEXITY_KEYWORDS.items():
        if any(_kw_in(kw, text) for kw in kws):
            complexity = level
            break

    desc_genre = matched_genre.replace("hiphop", "hip-hop").replace("dnb", "drum & bass")
    description = f"{desc_genre.title()} beat at {bpm} BPM, {bars} bars, {complexity} complexity"

    return {
        "genre":       matched_genre,
        "bpm":         bpm,
        "bars":        bars,
        "complexity":  complexity,
        "description": description,
    }


# ══════════════════════════════════════════════════════════════════════
# 2.  DRUM PATTERNS  (16-step grids, 1 = hit  0 = rest)
# ══════════════════════════════════════════════════════════════════════
# Keys: kick, snare, hihat_c (closed), hihat_o (open), clap

_PATTERNS: dict[str, dict[str, dict[str, list]]] = {
    "hiphop": {
        "simple": {
            "kick":    [1,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,1,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
            "hihat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
        "medium": {
            "kick":    [1,0,0,1, 0,0,1,0, 1,0,0,0, 0,1,0,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,1],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
        "complex": {
            "kick":    [1,0,1,0, 1,0,0,1, 0,1,0,0, 1,0,1,0],
            "snare":   [0,0,0,0, 1,0,0,1, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,1,1],
            "clap":    [0,0,1,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
        },
    },
    "trap": {
        "simple": {
            "kick":    [1,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
            "snare":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,1,0,1, 1,0,1,1, 0,1,1,0, 1,1,0,1],
            "hihat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0],
        },
        "medium": {
            "kick":    [1,0,0,1, 0,0,0,0, 1,0,0,0, 0,0,1,0],
            "snare":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,0, 0,1,0,0, 0,0,0,0, 0,1,0,0],
            "clap":    [0,0,0,0, 0,0,1,0, 0,0,0,0, 1,0,0,0],
        },
        "complex": {
            "kick":    [1,0,1,0, 0,1,0,0, 1,0,0,1, 0,0,1,0],
            "snare":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,1],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1],
            "clap":    [0,0,0,0, 0,0,1,0, 0,1,0,0, 1,0,0,0],
        },
    },
    "edm": {
        "simple": {
            "kick":    [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0],
            "hihat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
            "clap":    [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        },
        "medium": {
            "kick":    [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
            "hihat_o": [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,1],
            "clap":    [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        },
        "complex": {
            "kick":    [1,0,0,1, 1,0,0,0, 0,0,1,0, 1,0,0,0],
            "snare":   [0,0,0,0, 1,0,0,1, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1],
            "clap":    [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,1],
        },
    },
    "rock": {
        "simple": {
            "kick":    [1,0,0,0, 0,0,1,0, 1,0,0,0, 0,0,1,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
            "hihat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
        "medium": {
            "kick":    [1,0,0,0, 1,0,1,0, 1,0,0,0, 0,0,1,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
        "complex": {
            "kick":    [1,0,1,0, 1,0,0,1, 0,1,0,0, 1,0,1,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,1,0, 1,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,1,0],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
    },
    "metal": {
        "simple": {
            "kick":    [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
        "medium": {
            "kick":    [1,1,0,1, 0,1,1,0, 1,1,0,1, 0,1,0,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,1],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
        "complex": {
            "kick":    [1,1,1,0, 1,1,0,1, 1,0,1,1, 0,1,1,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
    },
    "jazz": {
        "simple": {
            "kick":    [1,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
            "snare":   [0,0,1,0, 0,0,0,0, 0,0,0,0, 0,1,0,0],
            "hihat_c": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
            "hihat_o": [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
        "medium": {
            "kick":    [1,0,0,0, 0,1,0,0, 0,0,0,1, 0,0,0,0],
            "snare":   [0,0,0,0, 0,0,1,0, 0,1,0,0, 0,0,0,1],
            "hihat_c": [1,1,0,1, 0,1,1,0, 1,0,1,1, 0,1,0,1],
            "hihat_o": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
        "complex": {
            "kick":    [1,0,0,1, 0,0,1,0, 0,1,0,0, 1,0,0,1],
            "snare":   [0,1,0,0, 0,0,0,1, 0,0,1,0, 0,0,0,0],
            "hihat_c": [1,1,0,1, 1,0,1,1, 0,1,1,0, 1,1,0,1],
            "hihat_o": [0,0,0,0, 0,1,0,0, 0,0,0,0, 0,0,1,0],
            "clap":    [0,0,1,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
        },
    },
    "reggae": {
        "simple": {
            "kick":    [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
            "snare":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,0,0,1, 0,0,1,0, 0,1,0,0, 1,0,0,1],
            "hihat_o": [0,0,1,0, 0,1,0,0, 0,0,0,1, 0,1,0,0],
            "clap":    [0,0,0,1, 0,0,0,0, 0,0,0,1, 0,0,0,0],
        },
        "medium": {
            "kick":    [1,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "snare":   [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
            "hihat_c": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
            "hihat_o": [0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,0,0],
            "clap":    [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,1],
        },
        "complex": {
            "kick":    [1,0,0,0, 1,0,0,1, 0,0,0,0, 1,0,0,0],
            "snare":   [0,0,0,0, 0,0,0,0, 1,0,0,1, 0,0,0,0],
            "hihat_c": [1,1,1,1, 0,1,1,0, 1,1,0,1, 1,0,1,1],
            "hihat_o": [0,0,0,1, 0,0,0,0, 0,0,1,0, 0,1,0,0],
            "clap":    [0,0,0,0, 0,1,0,0, 0,0,0,0, 0,0,1,0],
        },
    },
    "dnb": {
        "simple": {
            "kick":    [1,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
            "clap":    [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        },
        "medium": {
            "kick":    [1,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,1,0,0, 1,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,0, 0,1,0,0, 0,0,0,0, 0,1,0,0],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
        "complex": {
            "kick":    [1,0,0,0, 0,1,0,0, 1,0,0,1, 0,0,0,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,1,0, 1,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,0],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
    },
    "ambient": {
        "simple": {
            "kick":    [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
            "snare":   [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
            "hihat_c": [0,0,1,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
            "hihat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
        "medium": {
            "kick":    [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
            "snare":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,1,0],
            "hihat_c": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,0],
            "hihat_o": [0,0,0,1, 0,0,0,0, 0,0,0,0, 1,0,0,0],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
        "complex": {
            "kick":    [1,0,0,0, 0,0,1,0, 0,0,0,0, 0,1,0,0],
            "snare":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,1,0],
            "hihat_c": [1,0,0,1, 0,1,0,0, 0,0,1,0, 1,0,0,0],
            "hihat_o": [0,0,1,0, 0,0,0,0, 0,1,0,0, 0,0,0,1],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
    },
    "afrobeats": {
        "simple": {
            "kick":    [1,0,0,0, 0,0,1,0, 0,0,0,0, 1,0,0,0],
            "snare":   [0,0,1,0, 0,0,0,0, 0,1,0,0, 0,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,1],
            "clap":    [0,0,0,0, 0,1,0,0, 0,0,0,0, 0,1,0,0],
        },
        "medium": {
            "kick":    [1,0,0,1, 0,0,1,0, 0,0,1,0, 0,0,0,1],
            "snare":   [0,0,1,0, 0,0,0,0, 0,1,0,0, 0,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,1],
            "clap":    [0,0,0,0, 0,1,0,0, 0,0,0,0, 0,1,0,0],
        },
        "complex": {
            "kick":    [1,0,0,1, 0,1,1,0, 0,0,1,0, 1,0,0,1],
            "snare":   [0,1,0,0, 0,0,0,0, 0,1,0,1, 0,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1],
            "clap":    [0,0,0,0, 0,1,0,0, 1,0,0,0, 0,0,1,0],
        },
    },
    "funk": {
        "simple": {
            "kick":    [1,0,0,1, 0,0,0,0, 1,0,0,1, 0,0,0,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,1],
            "clap":    [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        },
        "medium": {
            "kick":    [1,0,1,0, 0,0,0,1, 0,0,1,0, 0,1,0,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,1, 1,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,1],
            "clap":    [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        },
        "complex": {
            "kick":    [1,0,1,0, 1,0,0,1, 0,1,0,0, 1,0,1,0],
            "snare":   [0,0,0,1, 1,0,0,0, 0,0,1,0, 1,0,0,1],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,0, 0,1,0,0, 0,0,0,0, 0,1,0,0],
            "clap":    [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        },
    },
    "latin": {
        "simple": {
            "kick":    [1,0,0,0, 0,0,1,0, 0,0,0,1, 0,0,0,0],
            "snare":   [0,0,1,0, 0,0,0,0, 0,0,0,0, 0,1,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,1, 0,0,0,0, 0,0,0,1, 0,0,0,0],
            "clap":    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        },
        "medium": {
            "kick":    [1,0,0,0, 0,1,0,0, 1,0,0,0, 0,0,1,0],
            "snare":   [0,0,1,0, 0,0,0,1, 0,0,1,0, 0,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,1, 0,0,0,0, 0,0,0,1, 0,0,0,1],
            "clap":    [0,0,0,0, 0,0,1,0, 0,0,0,0, 0,1,0,0],
        },
        "complex": {
            "kick":    [1,0,0,1, 0,0,1,0, 1,0,0,0, 0,1,0,0],
            "snare":   [0,0,1,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "hihat_o": [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,0],
            "clap":    [0,0,0,0, 0,1,0,0, 0,0,0,0, 0,1,0,0],
        },
    },
}

# Fallback for unknown genres
_FALLBACK_GENRE = "hiphop"


def get_pattern(genre: str, complexity: str) -> dict[str, list[int]]:
    """Return the 16-step drum pattern for a given genre / complexity."""
    genre_patterns = _PATTERNS.get(genre, _PATTERNS[_FALLBACK_GENRE])
    return genre_patterns.get(complexity, genre_patterns["medium"])


# ══════════════════════════════════════════════════════════════════════
# 3.  DRUM SOUND SYNTHESISERS  (pure numpy)
# ══════════════════════════════════════════════════════════════════════

def _env(length: int, attack: float = 0.002, decay: float = 0.15) -> np.ndarray:
    """ADSR-like amplitude envelope (only A+D phases relevant for one-shots)."""
    t = np.linspace(0, 1, length)
    env = np.exp(-t / max(decay, 1e-6))
    atk = int(attack * SR)
    if atk > 0:
        env[:atk] *= np.linspace(0, 1, atk)
    return env


def synthesize_kick(duration: float = 0.4) -> np.ndarray:
    """808-style kick: pitched sine with fast pitch sweep + click."""
    n = int(duration * SR)
    t = np.linspace(0, duration, n)
    # Pitch sweep 200→50 Hz
    freq = np.linspace(200, 50, n)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    sine = np.sin(phase)
    env = _env(n, attack=0.001, decay=0.25)
    # Transient click
    click_n = int(0.005 * SR)
    click = np.zeros(n)
    click[:click_n] = (np.random.rand(click_n) * 2 - 1) * np.linspace(1, 0, click_n)
    return (sine * env * 0.8 + click * 0.3).astype(np.float32)


def synthesize_snare(duration: float = 0.25) -> np.ndarray:
    """Snare: noise burst + pitched body."""
    n = int(duration * SR)
    t = np.linspace(0, duration, n)
    noise = np.random.randn(n)
    env_noise = _env(n, attack=0.001, decay=0.08)
    body = np.sin(2 * np.pi * 200 * t) * _env(n, attack=0.001, decay=0.05)
    return ((noise * env_noise * 0.6 + body * 0.4)).astype(np.float32)


def synthesize_hihat_closed(duration: float = 0.05) -> np.ndarray:
    """Closed hi-hat: very short filtered noise."""
    n = int(duration * SR)
    noise = np.random.randn(n)
    env = _env(n, attack=0.0005, decay=0.02)
    # simple HPF approximation via diff
    filtered = np.diff(noise, prepend=noise[0])
    return (filtered * env * 0.5).astype(np.float32)


def synthesize_hihat_open(duration: float = 0.25) -> np.ndarray:
    """Open hi-hat: longer filtered noise with metal ring."""
    n = int(duration * SR)
    noise = np.random.randn(n)
    env = _env(n, attack=0.001, decay=0.18)
    filtered = np.diff(noise, prepend=noise[0])
    ring = np.sin(2 * np.pi * 6000 * np.linspace(0, duration, n))
    return (filtered * env * 0.4 + ring * env * 0.1).astype(np.float32)


def synthesize_clap(duration: float = 0.12) -> np.ndarray:
    """Clap: layered noise bursts to mimic hand-clap."""
    n = int(duration * SR)
    noise = np.random.randn(n)
    env1 = _env(n, attack=0.001, decay=0.04)
    env2 = np.zeros(n)
    offset = int(0.01 * SR)
    if offset < n:
        env2[offset:] = _env(n - offset, attack=0.001, decay=0.03)
    return (noise * (env1 * 0.5 + env2 * 0.5) * 0.6).astype(np.float32)


# Pre-synthesise sounds once at module load
_SOUNDS: dict[str, np.ndarray] = {
    "kick":    synthesize_kick(),
    "snare":   synthesize_snare(),
    "hihat_c": synthesize_hihat_closed(),
    "hihat_o": synthesize_hihat_open(),
    "clap":    synthesize_clap(),
}


# ══════════════════════════════════════════════════════════════════════
# 4.  RENDERER
# ══════════════════════════════════════════════════════════════════════

def render_beat(
    pattern: dict[str, list[int]],
    bpm: float,
    bars: int,
    swing: float = 0.0,
) -> np.ndarray:
    """
    Render a 16-step pattern (one bar) repeated `bars` times.

    Args:
        pattern:  dict of instrument → 16-step list
        bpm:      beats per minute
        bars:     how many times to repeat the pattern
        swing:    0.0 = straight, 0.5 = full swing (delays odd 16th notes)

    Returns:
        Stereo float32 array shape (N, 2)
    """
    beat_dur = 60.0 / bpm                # quarter-note duration in seconds
    step_dur = beat_dur / 4.0            # 16th-note duration
    steps_per_bar = 16
    bar_samples = int(steps_per_bar * step_dur * SR)
    total_samples = bar_samples * bars

    stereo = np.zeros((total_samples, 2), dtype=np.float32)

    pan_map = {
        "kick":    (0.0,  0.0),    # centre
        "snare":   (0.0,  0.0),
        "hihat_c": (-0.3, 0.3),    # slight left lean
        "hihat_o": (0.2, -0.2),    # slight right lean
        "clap":    (0.0,  0.0),
    }

    for instrument, steps in pattern.items():
        sound = _SOUNDS.get(instrument)
        if sound is None:
            continue
        gain = {"kick": 0.95, "snare": 0.80, "hihat_c": 0.55,
                "hihat_o": 0.60, "clap": 0.70}.get(instrument, 0.7)
        pan_l, pan_r = pan_map.get(instrument, (0.0, 0.0))
        for bar in range(bars):
            for step, hit in enumerate(steps):
                if not hit:
                    continue
                swing_offset = 0
                if swing > 0 and step % 2 == 1:
                    swing_offset = int(swing * step_dur * SR)
                sample_pos = (
                    bar * bar_samples
                    + int(step * step_dur * SR)
                    + swing_offset
                )
                end = min(sample_pos + len(sound), total_samples)
                write_len = end - sample_pos
                if write_len <= 0:
                    continue
                snd = sound[:write_len] * gain
                stereo[sample_pos:end, 0] += snd * (0.5 + pan_l)
                stereo[sample_pos:end, 1] += snd * (0.5 + pan_r)

    # Soft limiting / normalise
    peak = np.abs(stereo).max()
    if peak > 0.9:
        stereo = stereo * (0.9 / peak)

    return stereo


# ══════════════════════════════════════════════════════════════════════
# 5.  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def generate_beat(prompt: str, output_path: str) -> dict:
    """
    Full pipeline: parse prompt → get pattern → render → save WAV.

    Returns::
        {
          "genre":       "hiphop",
          "bpm":         90.0,
          "bars":        4,
          "complexity":  "medium",
          "description": "...",
          "duration":    8.53,
          "pattern":     {"kick": [...], "snare": [...], ...},
        }
    """
    params = parse_prompt(prompt)
    logger.info(
        "Generating beat: genre=%s bpm=%.1f bars=%d complexity=%s",
        params["genre"], params["bpm"], params["bars"], params["complexity"],
    )

    pattern = get_pattern(params["genre"], params["complexity"])

    # Swing on jazz / funk / reggae feels natural
    swing = 0.25 if params["genre"] in ("jazz", "funk", "reggae") else 0.0

    audio = render_beat(pattern, bpm=params["bpm"], bars=params["bars"], swing=swing)
    duration = len(audio) / SR

    sf.write(output_path, audio, SR, subtype="PCM_16")
    logger.info("Beat saved to %s (%.2f s)", output_path, duration)

    return {
        **params,
        "duration": round(duration, 3),
        "pattern":  pattern,
        "sample_rate": SR,
    }
