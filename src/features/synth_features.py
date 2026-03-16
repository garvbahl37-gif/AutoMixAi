import librosa
import numpy as np

def extract_pitch_features(y, sr):
    """
    Extract pitch tracking features (e.g., f0) from audio.
    Used for tracking synth melodies.
    """
    # Use librosa.yin or pyin for fundamental frequency estimation
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    # Replace NaNs with zeros for unvoiced regions
    f0 = np.nan_to_num(f0)
    return {
        "mean_f0": np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0.0,
        "std_f0": np.std(f0[f0 > 0]) if np.any(f0 > 0) else 0.0,
        "voiced_duration_ratio": np.sum(voiced_flag) / len(voiced_flag)
    }

def extract_amplitude_features(y):
    """
    Extract amplitude envelope and energy features.
    """
    rms = librosa.feature.rms(y=y)[0]
    return {
        "mean_rms": np.mean(rms),
        "std_rms": np.std(rms),
        "max_rms": np.max(rms)
    }

def extract_bass_features(y, sr):
    """
    Extract low-frequency characteristics, isolating kick and bass synth bands.
    """
    # Simple low pass approximation using spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=20)
    # The first band corresponds to the lowest frequencies
    bass_band = contrast[0]
    return {
        "mean_bass_contrast": np.mean(bass_band),
        "std_bass_contrast": np.std(bass_band)
    }

def get_all_synth_features(y, sr):
    """
    Aggregate all synth-related features.
    """
    features = {}
    features.update(extract_pitch_features(y, sr))
    features.update(extract_amplitude_features(y))
    features.update(extract_bass_features(y, sr))
    return features
