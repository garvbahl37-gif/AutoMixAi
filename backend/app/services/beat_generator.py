"""
AutoMixAI – Advanced Beat Generator Service (v2)

Generates sophisticated drum beats from natural-language prompts using:
- Advanced NLP prompt parsing with synonyms and context understanding
- 20+ genres with sub-genre variations
- 10 drum instruments with realistic synthesis
- Humanization, swing, and velocity variation
- Fill patterns for transitions
- Time signature support (4/4, 3/4, 6/8, etc.)
- Dynamic pattern generation based on mood/energy
- MIDI pattern integration (when available)

Pipeline:
  1. parse_prompt()      — extract all parameters from any natural language
  2. generate_pattern()  — create or select pattern based on params
  3. synthesize_*()      — pure-numpy drum synthesis (10 instruments)
  4. render_beat()       — humanized rendering with variations
  5. generate_beat()     — public entry-point
"""

from __future__ import annotations

import re
import math
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import soundfile as sf

from app.utils.logger import get_logger

logger = get_logger(__name__)

SR = 44100  # internal sample rate


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA CLASSES & ENUMS
# ══════════════════════════════════════════════════════════════════════════════

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
    """All parameters extracted from a prompt."""
    genre: str
    sub_genre: Optional[str]
    bpm: float
    bars: int
    complexity: Complexity
    energy: Energy
    mood: Mood
    time_signature: Tuple[int, int]  # (numerator, denominator)
    swing: float  # 0.0 to 1.0
    humanize: float  # 0.0 to 1.0
    include_fills: bool
    instruments: List[str]
    description: str


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMPREHENSIVE KEYWORD MAPPINGS
# ══════════════════════════════════════════════════════════════════════════════

_GENRE_MAPPINGS: Dict[str, Dict[str, Any]] = {
    # Hip-Hop family
    "hiphop": {
        "keywords": ["hip hop", "hip-hop", "hiphop", "rap", "rapper", "rapping",
                     "mc", "emcee", "urban", "street"],
        "sub_genres": {
            "boom_bap": ["boom bap", "boom-bap", "boombap", "90s hip hop", "golden age",
                        "old school", "oldschool", "classic hip hop"],
            "lofi": ["lo-fi", "lofi", "lo fi", "chill hop", "chillhop", "study beats",
                    "relaxing hip hop", "jazzy hip hop"],
            "trap": ["trap", "atlanta", "southern trap", "hard trap"],
            "drill": ["drill", "uk drill", "chicago drill", "ny drill"],
            "crunk": ["crunk", "dirty south", "southern"],
        },
        "default_bpm": 90,
        "bpm_range": (70, 110),
    },
    "trap": {
        "keywords": ["trap", "trapbeat", "atlanta trap", "trap music"],
        "sub_genres": {
            "hard_trap": ["hard trap", "dark trap", "aggressive trap"],
            "melodic_trap": ["melodic trap", "emotional trap", "sad trap"],
            "plugg": ["plugg", "plug", "pluggnb"],
            "phonk": ["phonk", "drift phonk", "brazilian phonk"],
        },
        "default_bpm": 140,
        "bpm_range": (120, 160),
    },
    # Electronic family
    "edm": {
        "keywords": ["edm", "electronic dance", "dance music", "electronic"],
        "sub_genres": {
            "house": ["house", "deep house", "tech house", "progressive house",
                     "future house", "tropical house"],
            "techno": ["techno", "industrial techno", "minimal techno", "hard techno"],
            "trance": ["trance", "psytrance", "progressive trance", "uplifting trance"],
            "dubstep": ["dubstep", "brostep", "riddim"],
            "electro": ["electro", "electro house", "complextro"],
            "hardstyle": ["hardstyle", "hardcore", "gabber"],
        },
        "default_bpm": 128,
        "bpm_range": (118, 150),
    },
    "house": {
        "keywords": ["house", "house music", "4x4", "four on the floor"],
        "sub_genres": {
            "deep": ["deep house", "deep", "soulful house"],
            "tech": ["tech house", "tech", "minimal"],
            "progressive": ["progressive house", "prog house"],
            "disco": ["disco house", "nu disco", "french house"],
        },
        "default_bpm": 124,
        "bpm_range": (118, 130),
    },
    "techno": {
        "keywords": ["techno", "berlin techno", "detroit techno"],
        "sub_genres": {
            "industrial": ["industrial", "industrial techno", "hard"],
            "minimal": ["minimal techno", "minimal", "micro"],
            "acid": ["acid techno", "acid", "303"],
            "dub": ["dub techno", "dub"],
        },
        "default_bpm": 130,
        "bpm_range": (120, 145),
    },
    "dnb": {
        "keywords": ["drum and bass", "drum & bass", "dnb", "d&b", "d'n'b"],
        "sub_genres": {
            "liquid": ["liquid dnb", "liquid", "soulful dnb"],
            "jump_up": ["jump up", "jump-up", "dancefloor"],
            "neurofunk": ["neurofunk", "neuro", "tech dnb"],
            "jungle": ["jungle", "ragga jungle", "old school jungle"],
        },
        "default_bpm": 174,
        "bpm_range": (160, 180),
    },
    # Rock family
    "rock": {
        "keywords": ["rock", "rock music", "rock beat", "rock drums"],
        "sub_genres": {
            "classic": ["classic rock", "70s rock", "80s rock"],
            "indie": ["indie rock", "indie", "alternative"],
            "punk": ["punk", "punk rock", "pop punk"],
            "grunge": ["grunge", "seattle", "90s rock"],
            "hard_rock": ["hard rock", "heavy rock"],
        },
        "default_bpm": 120,
        "bpm_range": (100, 150),
    },
    "metal": {
        "keywords": ["metal", "heavy metal", "metal drums"],
        "sub_genres": {
            "thrash": ["thrash", "thrash metal", "speed metal"],
            "death": ["death metal", "death", "brutal"],
            "black": ["black metal", "black"],
            "prog": ["progressive metal", "prog metal", "djent"],
            "doom": ["doom", "doom metal", "sludge"],
        },
        "default_bpm": 160,
        "bpm_range": (100, 220),
    },
    "pop": {
        "keywords": ["pop", "pop music", "mainstream", "radio", "chart"],
        "sub_genres": {
            "dance_pop": ["dance pop", "europop"],
            "synth_pop": ["synth pop", "synthpop", "electropop"],
            "indie_pop": ["indie pop"],
            "k_pop": ["k-pop", "kpop", "korean pop"],
        },
        "default_bpm": 120,
        "bpm_range": (95, 135),
    },
    # Jazz & Blues family
    "jazz": {
        "keywords": ["jazz", "jazzy", "jazz drums", "swing"],
        "sub_genres": {
            "swing": ["swing", "big band", "traditional jazz"],
            "bebop": ["bebop", "bop", "hard bop"],
            "fusion": ["jazz fusion", "fusion", "jazz rock"],
            "latin_jazz": ["latin jazz", "afro-cuban jazz"],
            "smooth": ["smooth jazz", "contemporary jazz"],
        },
        "default_bpm": 120,
        "bpm_range": (60, 200),
    },
    "blues": {
        "keywords": ["blues", "bluesy", "12 bar", "shuffle"],
        "sub_genres": {
            "chicago": ["chicago blues", "electric blues"],
            "delta": ["delta blues", "acoustic blues"],
            "shuffle": ["blues shuffle", "shuffle beat"],
        },
        "default_bpm": 90,
        "bpm_range": (60, 130),
    },
    # World & Latin family
    "reggae": {
        "keywords": ["reggae", "roots reggae", "one drop"],
        "sub_genres": {
            "dub": ["dub", "dub reggae"],
            "ska": ["ska", "rocksteady"],
            "dancehall": ["dancehall", "ragga"],
            "reggaeton": ["reggaeton", "dembow"],
        },
        "default_bpm": 80,
        "bpm_range": (65, 100),
    },
    "latin": {
        "keywords": ["latin", "latino", "spanish"],
        "sub_genres": {
            "salsa": ["salsa", "mambo"],
            "bossa_nova": ["bossa nova", "bossa"],
            "samba": ["samba", "brazilian"],
            "cumbia": ["cumbia", "colombian"],
            "tango": ["tango", "milonga"],
        },
        "default_bpm": 100,
        "bpm_range": (80, 140),
    },
    "afrobeats": {
        "keywords": ["afrobeats", "afro", "african", "afropop"],
        "sub_genres": {
            "amapiano": ["amapiano", "piano", "south african"],
            "afro_house": ["afro house", "tribal house"],
            "highlife": ["highlife", "ghanaian"],
            "gqom": ["gqom", "durban"],
        },
        "default_bpm": 112,
        "bpm_range": (95, 125),
    },
    # Funk & Soul family
    "funk": {
        "keywords": ["funk", "funky", "groove", "groovy"],
        "sub_genres": {
            "classic": ["classic funk", "70s funk", "p-funk"],
            "disco": ["disco", "disco funk"],
            "soul": ["soul", "neo soul", "r&b"],
            "g_funk": ["g-funk", "west coast"],
        },
        "default_bpm": 105,
        "bpm_range": (85, 125),
    },
    # Ambient & Chill
    "ambient": {
        "keywords": ["ambient", "atmospheric", "soundscape"],
        "sub_genres": {
            "chill": ["chill", "chillout", "relaxing", "calm", "peaceful"],
            "downtempo": ["downtempo", "trip hop"],
            "meditation": ["meditation", "zen", "healing"],
            "dark_ambient": ["dark ambient", "drone"],
        },
        "default_bpm": 75,
        "bpm_range": (50, 100),
    },
    # Country & Folk
    "country": {
        "keywords": ["country", "country music", "western", "nashville"],
        "sub_genres": {
            "modern": ["modern country", "country pop"],
            "traditional": ["traditional country", "classic country"],
            "bluegrass": ["bluegrass", "folk"],
        },
        "default_bpm": 110,
        "bpm_range": (80, 140),
    },
    # Electronic Sub-genres as main
    "garage": {
        "keywords": ["garage", "uk garage", "2step", "2-step", "speed garage"],
        "sub_genres": {},
        "default_bpm": 130,
        "bpm_range": (125, 140),
    },
    "breakbeat": {
        "keywords": ["breakbeat", "breaks", "big beat", "nu skool breaks"],
        "sub_genres": {},
        "default_bpm": 130,
        "bpm_range": (110, 145),
    },
}

# Energy keywords
_ENERGY_KEYWORDS = {
    Energy.SOFT: ["soft", "gentle", "quiet", "subtle", "whisper", "delicate"],
    Energy.LOW: ["low energy", "relaxed", "laid-back", "laid back", "mellow", "easy"],
    Energy.MEDIUM: ["moderate", "balanced", "standard", "normal"],
    Energy.HIGH: ["high energy", "energetic", "powerful", "driving", "pumping", "upbeat"],
    Energy.INTENSE: ["intense", "aggressive", "hard", "heavy", "brutal", "extreme",
                     "banging", "raging", "insane"],
}

# Mood keywords
_MOOD_KEYWORDS = {
    Mood.DARK: ["dark", "sinister", "evil", "menacing", "ominous", "gloomy", "gothic"],
    Mood.MELANCHOLIC: ["sad", "melancholic", "emotional", "moody", "somber", "nostalgic",
                       "heartbreak", "lonely"],
    Mood.NEUTRAL: ["neutral", "balanced", "standard"],
    Mood.UPLIFTING: ["happy", "uplifting", "bright", "cheerful", "joyful", "positive",
                     "euphoric", "triumphant"],
    Mood.AGGRESSIVE: ["aggressive", "angry", "violent", "furious", "rage", "fierce"],
}

# Complexity keywords
_COMPLEXITY_KEYWORDS = {
    Complexity.MINIMAL: ["minimal", "minimalist", "sparse", "stripped", "bare"],
    Complexity.SIMPLE: ["simple", "basic", "straightforward", "clean", "easy"],
    Complexity.MEDIUM: ["medium", "moderate", "balanced", "standard"],
    Complexity.COMPLEX: ["complex", "busy", "filled", "intricate", "detailed"],
    Complexity.INTRICATE: ["very complex", "polyrhythmic", "polyrhythm", "syncopated",
                          "advanced", "technical", "progressive"],
}

# Instrument keywords
_INSTRUMENT_KEYWORDS = {
    "kick": ["kick", "bass drum", "bassdrum", "808"],
    "snare": ["snare", "snare drum", "backbeat"],
    "hihat_c": ["hi-hat", "hihat", "hi hat", "hats", "closed hat", "closed hi-hat"],
    "hihat_o": ["open hat", "open hi-hat", "open hihat"],
    "clap": ["clap", "handclap", "claps"],
    "rim": ["rim", "rimshot", "rim shot", "sidestick"],
    "tom": ["tom", "toms", "tom-tom", "floor tom"],
    "crash": ["crash", "crash cymbal"],
    "ride": ["ride", "ride cymbal"],
    "shaker": ["shaker", "tambourine", "maracas", "percussion"],
    "conga": ["conga", "congas", "bongo", "bongos"],
    "cowbell": ["cowbell", "bell"],
}

# Time signature keywords
_TIME_SIG_KEYWORDS = {
    (4, 4): ["4/4", "four four", "common time", "standard"],
    (3, 4): ["3/4", "three four", "waltz", "triple"],
    (6, 8): ["6/8", "six eight", "compound"],
    (5, 4): ["5/4", "five four", "odd time"],
    (7, 8): ["7/8", "seven eight", "odd"],
    (2, 4): ["2/4", "two four", "march", "polka"],
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. ADVANCED PROMPT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_text(text: str) -> str:
    """Normalize text for matching."""
    return text.lower().strip()


def _kw_in(kw: str, text: str) -> bool:
    """Case-insensitive word boundary match."""
    pattern = r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])"
    return bool(re.search(pattern, text, re.IGNORECASE))


def _find_genre(text: str) -> Tuple[str, Optional[str]]:
    """Find genre and sub-genre from text."""
    text = _normalize_text(text)

    # Sort genres by keyword length (longest first) to avoid partial matches
    all_matches = []
    for genre, data in _GENRE_MAPPINGS.items():
        for kw in data["keywords"]:
            if _kw_in(kw, text):
                all_matches.append((genre, None, len(kw)))

        # Check sub-genres
        for sub_genre, sub_kws in data.get("sub_genres", {}).items():
            for kw in sub_kws:
                if _kw_in(kw, text):
                    all_matches.append((genre, sub_genre, len(kw)))

    if all_matches:
        # Return longest match
        all_matches.sort(key=lambda x: x[2], reverse=True)
        return all_matches[0][0], all_matches[0][1]

    return "hiphop", None  # Default


def _find_bpm(text: str, genre: str) -> float:
    """Extract BPM from text or infer from context."""
    text = _normalize_text(text)

    # Explicit BPM patterns
    patterns = [
        r"(\d{2,3})\s*(?:bpm|tempo|beats?\s*per\s*min)",
        r"(?:bpm|tempo)[\s:]*(\d{2,3})",
        r"at\s*(\d{2,3})\s*(?:bpm)?",
        r"(\d{2,3})\s*bpm",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            bpm = float(match.group(1))
            return max(40.0, min(250.0, bpm))

    # Get default from genre
    genre_data = _GENRE_MAPPINGS.get(genre, {})
    default_bpm = genre_data.get("default_bpm", 120)
    bpm_range = genre_data.get("bpm_range", (80, 160))

    # Adjust based on energy/tempo words
    fast_words = ["fast", "quick", "rapid", "speedy", "uptempo", "energetic",
                  "high tempo", "racing", "frantic"]
    slow_words = ["slow", "downtempo", "relaxed", "laid-back", "chill",
                  "low tempo", "crawling", "sluggish"]
    very_fast = ["very fast", "super fast", "extremely fast", "insane tempo"]
    very_slow = ["very slow", "super slow", "extremely slow"]

    if any(_kw_in(w, text) for w in very_fast):
        return min(bpm_range[1], default_bpm * 1.3)
    elif any(_kw_in(w, text) for w in fast_words):
        return min(bpm_range[1], default_bpm * 1.15)
    elif any(_kw_in(w, text) for w in very_slow):
        return max(bpm_range[0], default_bpm * 0.7)
    elif any(_kw_in(w, text) for w in slow_words):
        return max(bpm_range[0], default_bpm * 0.85)

    return default_bpm


def _find_bars(text: str) -> int:
    """Extract bar count from text."""
    patterns = [
        r"(\d+)\s*(?:bar|bars|measure|measures)",
        r"(\d+)\s*(?:loop|loops|repeat|repeats)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            bars = int(match.group(1))
            return max(1, min(64, bars))

    # Check for duration-based requests
    duration_match = re.search(r"(\d+)\s*(?:second|sec|s)\b", text, re.IGNORECASE)
    if duration_match:
        # Rough estimate: assume 120 BPM, 4 beats per bar
        seconds = int(duration_match.group(1))
        bars = max(1, seconds // 2)
        return min(32, bars)

    return 4  # Default


def _find_enum_value(text: str, keywords_map: dict, default) -> Any:
    """Find enum value based on keyword matching."""
    text = _normalize_text(text)

    for enum_val, keywords in keywords_map.items():
        if any(_kw_in(kw, text) for kw in keywords):
            return enum_val

    return default


def _find_time_signature(text: str, genre: str) -> Tuple[int, int]:
    """Extract time signature from text."""
    text = _normalize_text(text)

    for time_sig, keywords in _TIME_SIG_KEYWORDS.items():
        if any(_kw_in(kw, text) for kw in keywords):
            return time_sig

    # Genre-based defaults
    waltz_genres = ["jazz", "country"]
    if genre in waltz_genres and "waltz" in text:
        return (3, 4)

    return (4, 4)  # Default


def _find_swing(text: str, genre: str) -> float:
    """Determine swing amount."""
    text = _normalize_text(text)

    if any(_kw_in(w, text) for w in ["no swing", "straight", "rigid", "quantized"]):
        return 0.0

    if any(_kw_in(w, text) for w in ["heavy swing", "full swing", "hard swing"]):
        return 0.7

    if any(_kw_in(w, text) for w in ["swing", "swung", "shuffle", "groove"]):
        return 0.4

    if any(_kw_in(w, text) for w in ["slight swing", "light swing", "subtle swing"]):
        return 0.2

    # Genre defaults
    swing_genres = {"jazz": 0.5, "blues": 0.4, "funk": 0.3, "reggae": 0.2}
    return swing_genres.get(genre, 0.0)


def _find_humanize(text: str) -> float:
    """Determine humanization amount."""
    text = _normalize_text(text)

    if any(_kw_in(w, text) for w in ["robotic", "quantized", "perfect", "machine", "precise"]):
        return 0.0

    if any(_kw_in(w, text) for w in ["human", "humanized", "organic", "natural", "live"]):
        return 0.6

    if any(_kw_in(w, text) for w in ["very human", "loose", "sloppy", "drunk"]):
        return 0.9

    return 0.3  # Default slight humanization


def _find_instruments(text: str, genre: str) -> List[str]:
    """Determine which instruments to include."""
    text = _normalize_text(text)

    requested = []
    excluded = []

    # Check for explicit includes/excludes
    for inst, keywords in _INSTRUMENT_KEYWORDS.items():
        for kw in keywords:
            if _kw_in(f"no {kw}", text) or _kw_in(f"without {kw}", text):
                excluded.append(inst)
            elif _kw_in(kw, text):
                requested.append(inst)

    # Base instruments by genre
    base_instruments = {
        "hiphop": ["kick", "snare", "hihat_c", "hihat_o", "clap"],
        "trap": ["kick", "snare", "hihat_c", "hihat_o", "clap"],
        "edm": ["kick", "snare", "hihat_c", "hihat_o", "clap", "crash"],
        "house": ["kick", "clap", "hihat_c", "hihat_o", "shaker"],
        "techno": ["kick", "hihat_c", "hihat_o", "clap", "rim"],
        "dnb": ["kick", "snare", "hihat_c", "hihat_o"],
        "rock": ["kick", "snare", "hihat_c", "hihat_o", "crash", "ride"],
        "metal": ["kick", "snare", "hihat_c", "tom", "crash", "ride"],
        "jazz": ["kick", "snare", "hihat_c", "hihat_o", "ride"],
        "reggae": ["kick", "snare", "hihat_c", "hihat_o", "rim"],
        "latin": ["kick", "snare", "hihat_c", "conga", "cowbell", "shaker"],
        "afrobeats": ["kick", "snare", "hihat_c", "shaker", "conga"],
        "funk": ["kick", "snare", "hihat_c", "hihat_o", "clap"],
        "ambient": ["kick", "hihat_c", "shaker"],
    }

    base = base_instruments.get(genre, ["kick", "snare", "hihat_c", "hihat_o"])

    # Combine
    if requested:
        instruments = list(set(requested + base))
    else:
        instruments = base

    # Apply exclusions
    instruments = [i for i in instruments if i not in excluded]

    return instruments if instruments else ["kick", "snare", "hihat_c"]


def _find_fills(text: str) -> bool:
    """Check if fills are requested."""
    text = _normalize_text(text)

    if any(_kw_in(w, text) for w in ["no fill", "no fills", "without fills"]):
        return False

    if any(_kw_in(w, text) for w in ["fill", "fills", "transition", "variation"]):
        return True

    return False  # Default no fills


def parse_prompt(prompt: str) -> BeatParams:
    """
    Advanced prompt parser that understands any natural language beat description.

    Examples it can understand:
    - "dark trap beat at 140 bpm, 8 bars, complex"
    - "make me a chill lo-fi hip hop loop with swing"
    - "aggressive metal drums, fast, 4/4"
    - "jazzy brush patterns with lots of swing"
    - "minimal techno kick and hi-hats, 130 bpm"
    - "latin percussion groove with congas"
    """
    text = _normalize_text(prompt)

    # Extract all parameters
    genre, sub_genre = _find_genre(text)
    bpm = _find_bpm(text, genre)
    bars = _find_bars(text)
    complexity = _find_enum_value(text, _COMPLEXITY_KEYWORDS, Complexity.MEDIUM)
    energy = _find_enum_value(text, _ENERGY_KEYWORDS, Energy.MEDIUM)
    mood = _find_enum_value(text, _MOOD_KEYWORDS, Mood.NEUTRAL)
    time_sig = _find_time_signature(text, genre)
    swing = _find_swing(text, genre)
    humanize = _find_humanize(text)
    instruments = _find_instruments(text, genre)
    fills = _find_fills(text)

    # Build description
    genre_name = genre.replace("_", " ").replace("dnb", "drum & bass").title()
    sub_name = f" ({sub_genre.replace('_', ' ')})" if sub_genre else ""
    desc = (f"{mood.value.title()} {energy.value} {genre_name}{sub_name} beat "
            f"at {bpm:.0f} BPM, {bars} bars, {complexity.value} complexity")

    return BeatParams(
        genre=genre,
        sub_genre=sub_genre,
        bpm=bpm,
        bars=bars,
        complexity=complexity,
        energy=energy,
        mood=mood,
        time_signature=time_sig,
        swing=swing,
        humanize=humanize,
        include_fills=fills,
        instruments=instruments,
        description=desc,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. PATTERN GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _get_steps_per_bar(time_sig: Tuple[int, int]) -> int:
    """Get number of 16th note steps for time signature."""
    num, denom = time_sig
    return int(num * (16 / denom))


def _generate_kick_pattern(params: BeatParams, steps: int) -> List[int]:
    """Generate kick drum pattern based on parameters."""
    pattern = [0] * steps

    # Base patterns by genre
    genre = params.genre
    complexity = params.complexity

    if genre in ["edm", "house", "techno"]:
        # Four-on-the-floor
        for i in range(0, steps, 4):
            pattern[i] = 1
        if complexity in [Complexity.COMPLEX, Complexity.INTRICATE]:
            # Add offbeats
            for i in [6, 14]:
                if i < steps:
                    pattern[i] = 1

    elif genre == "dnb":
        pattern[0] = 1
        if steps > 10:
            pattern[10] = 1
        if complexity in [Complexity.COMPLEX, Complexity.INTRICATE]:
            pattern[6] = 1

    elif genre in ["hiphop", "trap"]:
        pattern[0] = 1
        if complexity == Complexity.MINIMAL:
            pass
        elif complexity == Complexity.SIMPLE:
            if steps > 6:
                pattern[6] = 1
        else:
            if steps > 6:
                pattern[6] = 1
            if steps > 10:
                pattern[10] = 1
            if complexity == Complexity.INTRICATE:
                pattern[3] = 1
                if steps > 14:
                    pattern[14] = 1

    elif genre in ["rock", "metal"]:
        pattern[0] = 1
        if steps > 8:
            pattern[8] = 1
        if complexity in [Complexity.COMPLEX, Complexity.INTRICATE]:
            pattern[4] = 1
            if steps > 12:
                pattern[12] = 1

    elif genre == "reggae":
        pattern[0] = 1
        if complexity != Complexity.MINIMAL:
            if steps > 8:
                pattern[8] = 1

    elif genre == "jazz":
        pattern[0] = 1
        if steps > 10:
            pattern[10] = 1

    else:
        # Default pattern
        pattern[0] = 1
        if steps > 8:
            pattern[8] = 1

    return pattern


def _generate_snare_pattern(params: BeatParams, steps: int) -> List[int]:
    """Generate snare pattern."""
    pattern = [0] * steps
    genre = params.genre

    # Backbeat on 2 and 4 (steps 4 and 12 in 16th notes)
    if genre in ["rock", "pop", "funk", "hiphop"]:
        if steps > 4:
            pattern[4] = 1
        if steps > 12:
            pattern[12] = 1

    elif genre in ["edm", "house", "techno"]:
        if steps > 4:
            pattern[4] = 1
        if steps > 12:
            pattern[12] = 1

    elif genre == "dnb":
        if steps > 4:
            pattern[4] = 1
        if steps > 12:
            pattern[12] = 1
        if params.complexity in [Complexity.COMPLEX, Complexity.INTRICATE]:
            if steps > 10:
                pattern[10] = 1

    elif genre == "trap":
        if steps > 12:
            pattern[12] = 1  # Trap often has snare on 3

    elif genre == "reggae":
        # One-drop: snare on 3
        if steps > 8:
            pattern[8] = 1

    elif genre == "jazz":
        # Ghost notes and syncopation
        pattern[4] = 1
        if params.complexity != Complexity.MINIMAL:
            pattern[10] = 1

    else:
        if steps > 4:
            pattern[4] = 1
        if steps > 12:
            pattern[12] = 1

    return pattern


def _generate_hihat_pattern(params: BeatParams, steps: int, open_hat: bool = False) -> List[int]:
    """Generate hi-hat pattern."""
    pattern = [0] * steps
    genre = params.genre
    complexity = params.complexity

    if open_hat:
        # Open hats typically on upbeats
        if complexity == Complexity.MINIMAL:
            pass
        elif complexity == Complexity.SIMPLE:
            if steps > 7:
                pattern[7] = 1
        else:
            if steps > 7:
                pattern[7] = 1
            if steps > 15:
                pattern[15] = 1
    else:
        # Closed hats
        if complexity == Complexity.MINIMAL:
            for i in range(0, steps, 4):
                pattern[i] = 1
        elif complexity == Complexity.SIMPLE:
            for i in range(0, steps, 2):
                pattern[i] = 1
        elif complexity == Complexity.MEDIUM:
            for i in range(steps):
                pattern[i] = 1
        else:
            # Complex: all 16ths
            for i in range(steps):
                pattern[i] = 1

    # Genre-specific adjustments
    if genre == "trap" and not open_hat:
        # Trap hi-hat rolls
        for i in range(steps):
            pattern[i] = 1

    return pattern


def _generate_clap_pattern(params: BeatParams, steps: int) -> List[int]:
    """Generate clap pattern."""
    pattern = [0] * steps

    if params.genre in ["trap", "edm", "house"]:
        if steps > 4:
            pattern[4] = 1
        if steps > 12:
            pattern[12] = 1
    elif params.complexity in [Complexity.COMPLEX, Complexity.INTRICATE]:
        if steps > 4:
            pattern[4] = 1
        if steps > 12:
            pattern[12] = 1

    return pattern


def _generate_percussion_pattern(params: BeatParams, steps: int, inst: str) -> List[int]:
    """Generate pattern for percussion instruments."""
    pattern = [0] * steps

    if inst == "rim":
        # Rim shots on offbeats
        for i in range(2, steps, 4):
            pattern[i] = 1

    elif inst == "shaker":
        # Continuous shaker
        for i in range(0, steps, 2):
            pattern[i] = 1

    elif inst == "conga":
        # Latin pattern
        pattern[0] = 1
        if steps > 3:
            pattern[3] = 1
        if steps > 7:
            pattern[7] = 1
        if steps > 10:
            pattern[10] = 1

    elif inst == "cowbell":
        pattern[0] = 1
        if steps > 6:
            pattern[6] = 1
        if steps > 10:
            pattern[10] = 1

    elif inst in ["crash", "ride"]:
        # Cymbals
        if inst == "crash":
            pattern[0] = 1  # Crash on 1
        else:
            # Ride pattern
            for i in range(0, steps, 2):
                pattern[i] = 1

    elif inst == "tom":
        # Toms for fills
        if params.include_fills:
            if steps > 12:
                pattern[13] = 1
            if steps > 14:
                pattern[14] = 1
            if steps > 15:
                pattern[15] = 1

    return pattern


def generate_pattern(params: BeatParams) -> Dict[str, List[int]]:
    """Generate complete drum pattern based on parameters."""
    steps = _get_steps_per_bar(params.time_signature)
    pattern = {}

    for inst in params.instruments:
        if inst == "kick":
            pattern[inst] = _generate_kick_pattern(params, steps)
        elif inst == "snare":
            pattern[inst] = _generate_snare_pattern(params, steps)
        elif inst == "hihat_c":
            pattern[inst] = _generate_hihat_pattern(params, steps, open_hat=False)
        elif inst == "hihat_o":
            pattern[inst] = _generate_hihat_pattern(params, steps, open_hat=True)
        elif inst == "clap":
            pattern[inst] = _generate_clap_pattern(params, steps)
        else:
            pattern[inst] = _generate_percussion_pattern(params, steps, inst)

    return pattern


# ══════════════════════════════════════════════════════════════════════════════
# 5. DRUM SOUND SYNTHESIS (10 instruments)
# ══════════════════════════════════════════════════════════════════════════════

def _env(length: int, attack: float = 0.002, decay: float = 0.15) -> np.ndarray:
    """ADSR-like amplitude envelope."""
    t = np.linspace(0, 1, length)
    env = np.exp(-t / max(decay, 1e-6))
    atk = int(attack * SR)
    if atk > 0 and atk < length:
        env[:atk] *= np.linspace(0, 1, atk)
    return env


def synthesize_kick(duration: float = 0.4, pitch: float = 1.0, punch: float = 0.5) -> np.ndarray:
    """808-style kick with adjustable pitch and punch."""
    n = int(duration * SR)
    t = np.linspace(0, duration, n)

    base_freq = 55 * pitch
    freq = np.linspace(200 * pitch, base_freq, n)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    sine = np.sin(phase)
    env = _env(n, attack=0.001, decay=0.25)

    # Transient punch
    click_n = int(0.008 * SR)
    click = np.zeros(n)
    click[:click_n] = (np.random.rand(click_n) * 2 - 1) * np.linspace(1, 0, click_n)

    return (sine * env * 0.8 + click * punch * 0.4).astype(np.float32)


def synthesize_snare(duration: float = 0.25, tone: float = 200, snap: float = 0.6) -> np.ndarray:
    """Snare with adjustable tone and snap."""
    n = int(duration * SR)
    t = np.linspace(0, duration, n)

    noise = np.random.randn(n)
    env_noise = _env(n, attack=0.001, decay=0.08)
    body = np.sin(2 * np.pi * tone * t) * _env(n, attack=0.001, decay=0.05)

    return ((noise * env_noise * snap + body * (1 - snap * 0.5))).astype(np.float32)


def synthesize_hihat_closed(duration: float = 0.05) -> np.ndarray:
    """Closed hi-hat."""
    n = int(duration * SR)
    noise = np.random.randn(n)
    env = _env(n, attack=0.0005, decay=0.02)
    filtered = np.diff(noise, prepend=noise[0])
    return (filtered * env * 0.5).astype(np.float32)


def synthesize_hihat_open(duration: float = 0.3) -> np.ndarray:
    """Open hi-hat."""
    n = int(duration * SR)
    noise = np.random.randn(n)
    env = _env(n, attack=0.001, decay=0.2)
    filtered = np.diff(noise, prepend=noise[0])
    ring = np.sin(2 * np.pi * 6000 * np.linspace(0, duration, n))
    return (filtered * env * 0.4 + ring * env * 0.1).astype(np.float32)


def synthesize_clap(duration: float = 0.15) -> np.ndarray:
    """Hand clap."""
    n = int(duration * SR)
    noise = np.random.randn(n)

    # Multiple transients
    env = np.zeros(n)
    for offset_ms in [0, 10, 20, 30]:
        offset = int(offset_ms * SR / 1000)
        if offset < n:
            remaining = n - offset
            env[offset:] += _env(remaining, attack=0.001, decay=0.04) * (1 - offset_ms / 50)

    return (noise * env * 0.5).astype(np.float32)


def synthesize_rim(duration: float = 0.08) -> np.ndarray:
    """Rimshot/sidestick."""
    n = int(duration * SR)
    t = np.linspace(0, duration, n)

    click = np.sin(2 * np.pi * 1500 * t) * _env(n, attack=0.0001, decay=0.02)
    body = np.sin(2 * np.pi * 400 * t) * _env(n, attack=0.001, decay=0.04)

    return (click * 0.6 + body * 0.4).astype(np.float32)


def synthesize_tom(duration: float = 0.3, pitch: float = 100) -> np.ndarray:
    """Tom drum."""
    n = int(duration * SR)
    t = np.linspace(0, duration, n)

    freq = np.linspace(pitch * 1.5, pitch, n)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    sine = np.sin(phase)
    env = _env(n, attack=0.002, decay=0.2)

    return (sine * env * 0.7).astype(np.float32)


def synthesize_crash(duration: float = 1.0) -> np.ndarray:
    """Crash cymbal."""
    n = int(duration * SR)
    noise = np.random.randn(n)
    env = _env(n, attack=0.002, decay=0.6)

    # High frequency content
    t = np.linspace(0, duration, n)
    shimmer = np.sin(2 * np.pi * 8000 * t) + np.sin(2 * np.pi * 10000 * t)

    filtered = np.diff(noise, prepend=noise[0])
    return ((filtered * 0.4 + shimmer * 0.1) * env * 0.4).astype(np.float32)


def synthesize_ride(duration: float = 0.4) -> np.ndarray:
    """Ride cymbal."""
    n = int(duration * SR)
    noise = np.random.randn(n)
    env = _env(n, attack=0.001, decay=0.3)

    t = np.linspace(0, duration, n)
    ping = np.sin(2 * np.pi * 5000 * t) * _env(n, attack=0.0005, decay=0.05)

    filtered = np.diff(noise, prepend=noise[0])
    return ((filtered * env * 0.3 + ping * 0.4)).astype(np.float32)


def synthesize_shaker(duration: float = 0.08) -> np.ndarray:
    """Shaker/tambourine."""
    n = int(duration * SR)
    noise = np.random.randn(n)
    env = _env(n, attack=0.005, decay=0.05)

    # High-pass filter approximation
    filtered = np.diff(np.diff(noise, prepend=noise[0]), prepend=noise[0])
    return (filtered * env * 0.3).astype(np.float32)


def synthesize_conga(duration: float = 0.2, pitch: float = 200) -> np.ndarray:
    """Conga/bongo."""
    n = int(duration * SR)
    t = np.linspace(0, duration, n)

    freq = np.linspace(pitch * 1.2, pitch, n)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    sine = np.sin(phase)
    env = _env(n, attack=0.002, decay=0.15)

    # Add slap transient
    slap = np.random.randn(int(0.01 * SR))
    slap = np.pad(slap, (0, n - len(slap)))
    slap_env = _env(n, attack=0.0005, decay=0.02)

    return (sine * env * 0.6 + slap * slap_env * 0.3).astype(np.float32)


def synthesize_cowbell(duration: float = 0.15) -> np.ndarray:
    """Cowbell."""
    n = int(duration * SR)
    t = np.linspace(0, duration, n)

    f1, f2 = 800, 540
    tone = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    env = _env(n, attack=0.001, decay=0.1)

    return (tone * env * 0.4).astype(np.float32)


# Sound bank
_SOUNDS: Dict[str, np.ndarray] = {}


def _init_sounds():
    """Initialize sound bank."""
    global _SOUNDS
    _SOUNDS = {
        "kick": synthesize_kick(),
        "snare": synthesize_snare(),
        "hihat_c": synthesize_hihat_closed(),
        "hihat_o": synthesize_hihat_open(),
        "clap": synthesize_clap(),
        "rim": synthesize_rim(),
        "tom": synthesize_tom(),
        "crash": synthesize_crash(),
        "ride": synthesize_ride(),
        "shaker": synthesize_shaker(),
        "conga": synthesize_conga(),
        "cowbell": synthesize_cowbell(),
    }


_init_sounds()


# ══════════════════════════════════════════════════════════════════════════════
# 6. RENDERER WITH HUMANIZATION
# ══════════════════════════════════════════════════════════════════════════════

def render_beat(
    pattern: Dict[str, List[int]],
    params: BeatParams,
) -> np.ndarray:
    """
    Render pattern with humanization, swing, and velocity variation.
    """
    bpm = params.bpm
    bars = params.bars
    swing = params.swing
    humanize = params.humanize
    time_sig = params.time_signature

    steps_per_bar = _get_steps_per_bar(time_sig)
    beat_dur = 60.0 / bpm
    step_dur = beat_dur / 4.0
    bar_samples = int(steps_per_bar * step_dur * SR)
    total_samples = bar_samples * bars

    stereo = np.zeros((total_samples, 2), dtype=np.float32)

    # Panning
    pan_map = {
        "kick": 0.0, "snare": 0.0, "hihat_c": -0.3, "hihat_o": 0.25,
        "clap": 0.0, "rim": 0.1, "tom": 0.2, "crash": -0.4,
        "ride": 0.35, "shaker": -0.2, "conga": 0.3, "cowbell": -0.25,
    }

    # Base gains
    gain_map = {
        "kick": 0.95, "snare": 0.80, "hihat_c": 0.50, "hihat_o": 0.55,
        "clap": 0.70, "rim": 0.60, "tom": 0.75, "crash": 0.50,
        "ride": 0.55, "shaker": 0.40, "conga": 0.60, "cowbell": 0.55,
    }

    for instrument, steps in pattern.items():
        sound = _SOUNDS.get(instrument)
        if sound is None:
            continue

        base_gain = gain_map.get(instrument, 0.6)
        pan = pan_map.get(instrument, 0.0)

        for bar in range(bars):
            for step, hit in enumerate(steps):
                if not hit:
                    continue

                # Humanize timing
                timing_offset = 0
                if humanize > 0:
                    timing_offset = int(np.random.normal(0, humanize * 0.01 * SR))

                # Swing (delay odd 16th notes)
                swing_offset = 0
                if swing > 0 and step % 2 == 1:
                    swing_offset = int(swing * step_dur * SR * 0.5)

                # Velocity variation
                vel_variation = 1.0
                if humanize > 0:
                    vel_variation = 1.0 + np.random.uniform(-0.15, 0.1) * humanize

                # Accent on strong beats
                accent = 1.0
                if step == 0:
                    accent = 1.1
                elif step in [4, 8, 12]:
                    accent = 1.05

                sample_pos = (
                    bar * bar_samples
                    + int(step * step_dur * SR)
                    + swing_offset
                    + timing_offset
                )

                sample_pos = max(0, min(sample_pos, total_samples - 1))
                end = min(sample_pos + len(sound), total_samples)
                write_len = end - sample_pos

                if write_len <= 0:
                    continue

                gain = base_gain * vel_variation * accent
                snd = sound[:write_len] * gain

                # Stereo panning
                left_gain = 0.5 - pan * 0.5
                right_gain = 0.5 + pan * 0.5

                stereo[sample_pos:end, 0] += snd * left_gain
                stereo[sample_pos:end, 1] += snd * right_gain

    # Soft limiting
    peak = np.abs(stereo).max()
    if peak > 0.95:
        stereo = stereo * (0.95 / peak)

    return stereo


# ══════════════════════════════════════════════════════════════════════════════
# 7. PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def generate_beat(prompt: str, output_path: str) -> dict:
    """
    Generate a drum beat from any natural language prompt.

    Examples:
    - "dark trap beat 140 bpm"
    - "chill lo-fi hip hop"
    - "aggressive metal drums with double kick"
    - "jazzy swing beat with brushes"
    - "minimal techno 130 bpm, 8 bars"
    - "latin percussion groove with congas and cowbell"
    - "simple rock beat for practice"

    Returns dict with all parameters and pattern.
    """
    params = parse_prompt(prompt)

    logger.info(
        "Generating beat: genre=%s bpm=%.1f bars=%d complexity=%s energy=%s",
        params.genre, params.bpm, params.bars,
        params.complexity.value, params.energy.value,
    )

    pattern = generate_pattern(params)
    audio = render_beat(pattern, params)
    duration = len(audio) / SR

    sf.write(output_path, audio, SR, subtype="PCM_16")
    logger.info("Beat saved to %s (%.2f s)", output_path, duration)

    return {
        "genre": params.genre,
        "sub_genre": params.sub_genre,
        "bpm": params.bpm,
        "bars": params.bars,
        "complexity": params.complexity.value,
        "energy": params.energy.value,
        "mood": params.mood.value,
        "time_signature": f"{params.time_signature[0]}/{params.time_signature[1]}",
        "swing": params.swing,
        "humanize": params.humanize,
        "instruments": params.instruments,
        "description": params.description,
        "duration": round(duration, 3),
        "pattern": pattern,
        "sample_rate": SR,
    }
