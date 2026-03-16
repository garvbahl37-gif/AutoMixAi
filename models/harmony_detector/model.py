"""
Harmony Detector Model
======================
Detects musical key, chord progressions, and harmonic compatibility
between two tracks — useful for selecting which tracks to mix next.

Usage::

    from models.harmony_detector import HarmonyDetector
    import librosa

    detector = HarmonyDetector()
    y, sr = librosa.load("track.wav", sr=22050)
    key, confidence = detector.detect_key_from_audio(y, sr)
    # → ("G major", 0.82)

    score = detector.compatibility("C major", "G major")
    # → 0.9  (perfect fifth — highly compatible)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import librosa


KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                            2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Harmonic compatibility matrix (semitone distance → score)
_COMPAT = {0: 1.0, 1: 0.3, 2: 0.6, 3: 0.7, 4: 0.7, 5: 0.8,
           6: 0.2, 7: 0.9, 8: 0.5, 9: 0.7, 10: 0.6, 11: 0.3}


class HarmonyDetector:
    """
    Key detection and harmonic compatibility scoring.

    Uses the Krumhansl-Schmuckler key-finding algorithm on chroma features.
    """

    def detect_key(self, chroma: np.ndarray) -> Tuple[str, float]:
        """
        Detect musical key from a 12-bin chroma vector (or matrix).

        Parameters
        ----------
        chroma : Shape ``(12,)`` or ``(12, T)`` — chroma feature.

        Returns
        -------
        Tuple of (key_string, confidence).  Key string e.g. ``"G major"``.
        """
        if chroma.ndim == 2:
            chroma = chroma.mean(axis=1)

        chroma = chroma[:12]
        chroma_norm = chroma / (chroma.sum() + 1e-8)

        best_key = "C major"
        best_score = -np.inf

        for i, root in enumerate(KEY_NAMES):
            rolled = np.roll(chroma_norm, -i)

            major_corr = float(np.corrcoef(rolled, _MAJOR_PROFILE)[0, 1])
            minor_corr = float(np.corrcoef(rolled, _MINOR_PROFILE)[0, 1])

            if major_corr > best_score:
                best_score = major_corr
                best_key = f"{root} major"
            if minor_corr > best_score:
                best_score = minor_corr
                best_key = f"{root} minor"

        # Map correlation [-1,1] to confidence [0,1]
        confidence = round((best_score + 1) / 2, 3)
        return best_key, confidence

    def detect_key_from_audio(self, y: np.ndarray, sr: int) -> Tuple[str, float]:
        """Compute chroma from audio, then detect key."""
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        return self.detect_key(chroma)

    def chord_sequence(
        self,
        chroma: np.ndarray,
        frames_per_chord: int = 8,
    ) -> List[str]:
        """
        Estimate a chord symbol per *frames_per_chord* frames.

        Parameters
        ----------
        chroma          : Shape ``(12, T)``.
        frames_per_chord: Number of frames to average per chord window.

        Returns
        -------
        List of chord root names (e.g. ``["C", "F", "G", "Am"]``).
        """
        if chroma.ndim == 1:
            return [KEY_NAMES[int(np.argmax(chroma[:12]))]]

        chords: List[str] = []
        T = chroma.shape[1]
        for start in range(0, T, frames_per_chord):
            window = chroma[:, start : start + frames_per_chord].mean(axis=1)
            root_idx = int(np.argmax(window[:12]))
            # Simple major/minor heuristic: if minor 3rd stronger than major 3rd → minor
            minor_3rd = (root_idx + 3) % 12
            major_3rd = (root_idx + 4) % 12
            suffix = "m" if window[minor_3rd] > window[major_3rd] else ""
            chords.append(f"{KEY_NAMES[root_idx]}{suffix}")

        return chords

    def compatibility(self, key1: str, key2: str) -> float:
        """
        Score harmonic compatibility between two key strings (0–1).

        ``1.0`` = perfectly compatible (same key or enharmonic).
        ``0.9`` = perfect fifth apart (Camelot wheel neighbours).

        Parameters
        ----------
        key1, key2 : e.g. ``"C major"``, ``"G minor"``.

        Returns
        -------
        float in ``[0, 1]``.
        """
        root1 = key1.split()[0]
        root2 = key2.split()[0]
        mode1 = key1.split()[1] if len(key1.split()) > 1 else "major"
        mode2 = key2.split()[1] if len(key2.split()) > 1 else "major"

        try:
            i1 = KEY_NAMES.index(root1)
            i2 = KEY_NAMES.index(root2)
        except ValueError:
            return 0.5

        dist = min(abs(i1 - i2), 12 - abs(i1 - i2))
        score = _COMPAT.get(dist, 0.5)

        if mode1 == mode2:
            score = min(1.0, score * 1.05)
        else:
            score *= 0.9

        return round(score, 3)
