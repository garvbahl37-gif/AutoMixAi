import librosa
import numpy as np

def extract_drum_features(y, sr):
    """
    Extract features useful for distinguishing percussive FL Studio samples
    like Claps, Snares, Kicks, and Hi-hats.
    """
    # Zero crossing rate (high for snares/claps/hats, low for kicks)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    # Spectral flatness (high for noise-like sounds like snares/claps)
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    
    # Onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    return {
        "mean_zcr": np.mean(zcr),
        "std_zcr": np.std(zcr),
        "mean_centroid": np.mean(centroid),
        "mean_flatness": np.mean(flatness),
        "max_onset_strength": np.max(onset_env) if len(onset_env) > 0 else 0.0
    }
