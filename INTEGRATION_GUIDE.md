# GiantstepsKey Model Integration Guide

This document explains how to integrate the trained GiantstepsKey key detection model into the AutoMixAI backend.

## Quick Start

### 1. After Training Completes

The notebook outputs files to `kaggle/output/`:

```
output/
├── models/
│   ├── key_detector_giantsteps.h5              ← Download this
│   ├── key_label_encoder.pkl                   ← Download this
│   └── feature_normalization.pkl               ← Download this
├── data/
│   └── giantsteps_key_features.csv             ← Training data (optional)
└── combined_dataset_info.json
```

### 2. Copy Model to Backend

```bash
# From project root
cp kaggle/output/models/key_detector_giantsteps.h5 backend/app/storage/models/
cp kaggle/output/models/key_label_encoder.pkl backend/app/storage/models/
cp kaggle/output/models/feature_normalization.pkl backend/app/storage/models/
```

### 3. Create Inference Module

Create a new file: `backend/app/services/key_detector.py`

```python
"""
Musical Key Detection Service using GiantstepsKey Model

Detects the musical key (e.g., "C major", "D minor") from audio files
using the trained ANN model.
"""

import pickle
from pathlib import Path
import numpy as np
import librosa
import tensorflow as tf
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 13
FEATURE_DIM = 43

# Model paths
MODELS_DIR = Path(__file__).parent.parent / "storage" / "models"
MODEL_PATH = MODELS_DIR / "key_detector_giantsteps.h5"
ENCODER_PATH = MODELS_DIR / "key_label_encoder.pkl"
NORMALIZER_PATH = MODELS_DIR / "feature_normalization.pkl"


class KeyDetector:
    """Detects musical key from audio files."""
    
    def __init__(self):
        """Initialize model and preprocessing state."""
        self.model = None
        self.encoder = None
        self.normalizer = None
        self._load_model()
    
    def _load_model(self):
        """Load trained model and preprocessing state."""
        try:
            # Load Keras model
            self.model = tf.keras.models.load_model(str(MODEL_PATH))
            logger.info(f"✅ Loaded key detection model from {MODEL_PATH}")
            
            # Load label encoder
            with open(ENCODER_PATH, 'rb') as f:
                self.encoder = pickle.load(f)
            logger.info(f"✅ Loaded label encoder with {len(self.encoder.classes_)} classes")
            
            # Load feature normalizer
            with open(NORMALIZER_PATH, 'rb') as f:
                self.normalizer = pickle.load(f)
            logger.info(f"✅ Loaded feature normalizer")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Model files not found: {e}")
            logger.error(f"   Expected files:")
            logger.error(f"     • {MODEL_PATH}")
            logger.error(f"     • {ENCODER_PATH}")
            logger.error(f"     • {NORMALIZER_PATH}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def extract_features(self, y, sr=SAMPLE_RATE):
        """
        Extract 43-dimensional feature vector from audio.
        
        Args:
            y: Audio time series (numpy array)
            sr: Sample rate (default: 22050)
        
        Returns:
            features: 1D numpy array of shape (43,)
        """
        hop = HOP_LENGTH
        n_fft = N_FFT
        
        # STFT
        S_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        S_mag = np.abs(S_complex)
        
        # MFCCs and deltas
        mel = librosa.feature.melspectrogram(S=S_mag**2, sr=sr, n_fft=n_fft)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)
        delta_mfcc = librosa.feature.delta(mfcc)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(S=S_mag**2, sr=sr, hop_length=hop, n_fft=n_fft)
        
        # Onset, flux, energy, centroid, ZCR
        onset = librosa.onset.onset_strength(
            S=librosa.amplitude_to_db(S_mag, ref=np.max), sr=sr, hop_length=hop
        )
        flux = np.sqrt(np.sum(np.diff(S_mag, axis=1) ** 2, axis=0))
        flux = np.concatenate([[0.0], flux])
        rms = librosa.feature.rms(S=S_mag, frame_length=n_fft, hop_length=hop)[0]
        centroid = librosa.feature.spectral_centroid(S=S_mag, sr=sr, hop_length=hop, n_fft=n_fft)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0]
        
        # Combine and take means
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(delta_mfcc, axis=1),
            np.mean(chroma, axis=1),
            [np.mean(onset)],
            [np.mean(flux)],
            [np.mean(rms)],
            [np.mean(centroid)],
            [np.mean(zcr)],
        ])
        
        return features.astype(np.float32)
    
    def detect_key(self, audio_path, return_confidence=True):
        """
        Detect musical key from audio file.
        
        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            return_confidence: If True, return (key, confidence); else just key
        
        Returns:
            If return_confidence=True:
                dict: {
                    'key': 'C major',
                    'confidence': 0.95,
                    'top_3': [
                        {'key': 'C major', 'confidence': 0.95},
                        {'key': 'A minor', 'confidence': 0.03},
                        {'key': 'G major', 'confidence': 0.01},
                    ]
                }
            Else:
                str: 'C major'
        """
        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
            
            # Extract features
            features = self.extract_features(y, sr)
            
            # Normalize
            features_norm = (features - self.normalizer['mean']) / self.normalizer['std']
            
            # Predict
            prediction = self.model.predict(
                features_norm.reshape(1, -1),
                verbose=0
            )
            probs = prediction[0]
            
            # Get top predictions
            top_indices = np.argsort(probs)[::-1][:3]
            top_predictions = [
                {
                    'key': self.encoder.classes_[idx],
                    'confidence': float(probs[idx])
                }
                for idx in top_indices
            ]
            
            if return_confidence:
                return {
                    'key': top_predictions[0]['key'],
                    'confidence': top_predictions[0]['confidence'],
                    'top_3': top_predictions
                }
            else:
                return top_predictions[0]['key']
        
        except FileNotFoundError:
            logger.error(f"❌ Audio file not found: {audio_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error detecting key: {e}")
            raise


# Global instance
_key_detector = None


def get_key_detector():
    """Get or initialize the KeyDetector singleton."""
    global _key_detector
    if _key_detector is None:
        _key_detector = KeyDetector()
    return _key_detector
```

### 4. Add Route Endpoint

Create or update `backend/app/routes/analyze.py`:

```python
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.key_detector import get_key_detector
from app.schemas.analysis_response import AnalysisResponse
from app.utils.logger import get_logger
import tempfile
from pathlib import Path

router = APIRouter(prefix="/api", tags=["Analysis"])
logger = get_logger(__name__)


@router.post("/detect-key")
async def detect_key(file: UploadFile = File(...)):
    """
    Detect musical key from uploaded audio file.
    
    Returns:
        {
            'key': 'C major',
            'confidence': 0.95,
            'top_3': [
                {'key': 'C major', 'confidence': 0.95},
                {'key': 'A minor', 'confidence': 0.03},
                {'key': 'G major', 'confidence': 0.01},
            ]
        }
    """
    try:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        try:
            # Detect key
            detector = get_key_detector()
            result = detector.detect_key(tmp_path, return_confidence=True)
            
            logger.info(f"✅ Key detected: {result['key']} (confidence: {result['confidence']:.2%})")
            return result
        
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    
    except Exception as e:
        logger.error(f"❌ Key detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 5. Update Main App

In `backend/app/main.py`, register the route:

```python
from app.routes import analyze

app.include_router(analyze.router)
```

## Testing the Integration

### Test with Python

```python
from app.services.key_detector import get_key_detector

detector = get_key_detector()
result = detector.detect_key("path/to/audio.wav", return_confidence=True)
print(f"Key: {result['key']} ({result['confidence']:.1%})")
print(f"Top 3: {result['top_3']}")
```

### Test with API

```bash
curl -X POST http://localhost:8000/api/detect-key \
  -F "file=@path/to/audio.wav"
```

Expected response:
```json
{
  "key": "A minor",
  "confidence": 0.87,
  "top_3": [
    {"key": "A minor", "confidence": 0.87},
    {"key": "C major", "confidence": 0.08},
    {"key": "E minor", "confidence": 0.04}
  ]
}
```

## Performance Considerations

- **Inference time**: ~500-1000ms per track (depends on audio length)
- **Memory**: ~200 MB (model + librosa)
- **Batch inference**: Modify `detect_key()` to accept multiple audio paths

### Optimize for Speed

```python
def detect_key_batch(audio_paths, return_confidence=True):
    """Detect keys for multiple audio files (faster)."""
    predictions = []
    
    for path in audio_paths:
        # Load and preprocess
        y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        features = self.extract_features(y, sr)
        features_norm = (features - self.normalizer['mean']) / self.normalizer['std']
        predictions.append(features_norm)
    
    # Batch predict
    X = np.array(predictions)
    probs = self.model.predict(X, verbose=0)  # Much faster than loop
    
    # Return results
    return [...]
```

## Troubleshooting

### Error: "Model files not found"
- Ensure model files are copied to `backend/app/storage/models/`
- Check file paths match exactly

### Error: "Feature dimension mismatch"
- Ensure `FEATURE_DIM=43` in both training notebook and inference code
- Verify feature extraction code is identical

### Low accuracy on your audio
- GiantstepsKey is trained on electronic dance music (EDM) and Beatport previews
- May not generalize well to other genres
- Consider retraining on your target domain

## Next Steps

1. **Train multi-genre model**: Combine GiantstepsKey + Ballroom + other datasets
2. **Add confidence threshold**: Only return predictions > 0.8 confidence
3. **Cache predictions**: Store key info in database for repeated queries
4. **Visualize results**: Show confidence scores in frontend

