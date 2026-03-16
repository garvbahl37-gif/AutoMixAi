# AutoMixAI Combined Training Notebook - README

## Overview

**`AutoMixAI_Combined_Training.ipynb`** is a comprehensive Jupyter notebook that trains machine learning models on all available AutoMixAI datasets in a single, unified pipeline.

## What It Does

This notebook:
1. **Loads 4 datasets** with 1,878+ samples total
2. **Extracts audio features** (MFCC, spectral, chroma, etc.)
3. **Trains 2 models**:
   - **Beat Detection**: Binary classifier (beat/non-beat at frame level)
   - **Key Detection**: 24-class classifier (12 pitches × 2 modes)
4. **Evaluates performance** with comprehensive metrics
5. **Exports models** ready for production deployment
6. **Generates Kaggle submission** with predictions

## Datasets Used

| Dataset | Tracks | Purpose | Status |
|---------|--------|---------|--------|
| **Ballroom** | 698 | Beat detection | ✓ Integrated |
| **FMA Small** | ~850 | Beat detection | ✓ Integrated |
| **MedleyDB** | 330+ | Beat detection + instruments | ✓ Integrated |
| **GiantstepsKey** | 604 | Key detection | ✓ Integrated |
| **TOTAL** | **2,482+** | Combined training | **✓ Ready** |

## Notebook Sections

### 1. **Import Required Libraries** (Section 1)
- Data processing: pandas, numpy
- Audio processing: librosa, scipy, soundfile
- Machine learning: sklearn, TensorFlow/Keras
- Visualization: matplotlib, seaborn

```python
import tensorflow as tf
from sklearn.model_selection import train_test_split
import librosa
import pandas as pd
```

### 2. **Load All Datasets** (Section 2)
- Configures paths to all 4 datasets
- Verifies dataset availability
- Implements feature extraction functions
- **Ballroom loader**: WAV files + BPM annotations
- **FMA loader**: MP3 files + metadata CSV
- **MedleyDB loader**: YAML metadata + mix audio
- **GiantstepsKey loader**: Audio + key annotations

**Output**: 4 feature matrices (X) and label vectors (y)

### 3. **Combine and Preprocess** (Section 3)
- Combines Ballroom + FMA + MedleyDB for beat detection
- Keeps GiantstepsKey separate (different task)
- Standardizes features using StandardScaler
- Saves scalers for later inference

```python
X_beat_combined = np.vstack([X_ballroom, X_fma, X_medleydb])
scaler = StandardScaler().fit(X_beat_combined)
X_beat_scaled = scaler.transform(X_beat_combined)
```

### 4. **Split Data** (Section 4)
- 80% training / 20% testing split
- Stratified split for balanced distribution
- Beat data: Frame-level labels
- Key data: Track-level labels

```python
X_beat_train, X_beat_test, y_beat_train, y_beat_test = train_test_split(
    X_beat_scaled, y_beat_combined,
    test_size=0.2, stratify=np.round(y_beat_combined)
)
```

### 5. **Train Models** (Section 5)

#### Beat Detection Model
```
Input (32 features)
  ↓
Dense(256) + BatchNorm + Dropout(0.3)
  ↓
Dense(128) + BatchNorm + Dropout(0.3)
  ↓
Dense(64) + BatchNorm + Dropout(0.2)
  ↓
Dense(32) + Dropout(0.1)
  ↓
Dense(1, sigmoid) → Output [0-1] (beat probability)
```

#### Key Detection Model
```
Input (32 features)
  ↓
Dense(256) + BatchNorm + Dropout(0.3)
  ↓
Dense(128) + BatchNorm + Dropout(0.3)
  ↓
Dense(64) + BatchNorm + Dropout(0.2)
  ↓
Dense(32) + Dropout(0.1)
  ↓
Dense(24, softmax) → Output [0-1] x 24 (key probabilities)
```

Training parameters:
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss**: Binary crossentropy (beat), Sparse categorical crossentropy (key)
- **Epochs**: 20 (with early stopping)
- **Batch size**: 64 (beat), 32 (key)
- **Metrics**: Accuracy, AUC, Top-3 accuracy

### 6. **Make Predictions** (Section 6)
- Evaluates models on test sets
- Computes metrics: Accuracy, precision, recall, F1-score
- Generates confidence scores

### 7. **Generate Kaggle Submission** (Section 7)
- Creates `kaggle_submission.csv` with predictions
- Exports trained models (.h5 files)
- Saves feature scalers (.pkl files)
- Generates comprehensive training report

## Output Files

Generated in `notebooks/kaggle/output/`:

| File | Purpose |
|------|---------|
| `beat_detection_model.h5` | Trained beat detection model |
| `key_detection_model.h5` | Trained key detection model |
| `scaler_beat.pkl` | Feature scaler for beat data |
| `scaler_key.pkl` | Feature scaler for key data |
| `kaggle_submission.csv` | Test predictions with confidence |
| `training_summary.json` | Metadata and metrics (JSON) |
| `training_report.md` | Detailed training report (Markdown) |

## How to Use

### Option 1: Kaggle Kernel (Recommended)

1. **Upload datasets to Kaggle** (if not already available):
   - Ballroom
   - FMA Small
   - MedleyDB
   - GiantstepsKey

2. **Create new Kaggle notebook**:
   - Go to kaggle.com → Notebooks → New Notebook
   - Add datasets as input
   - Select **TPU** or **GPU** runtime for faster training

3. **Copy notebook code** into Kaggle notebook

4. **Run all cells** (kernel will handle the rest)

5. **Download outputs** from the Results tab

### Option 2: Local Jupyter

```bash
# Navigate to project directory
cd AutoMixAI

# Install dependencies (if needed)
pip install -r backend/requirements.txt

# Start Jupyter
jupyter notebook

# Open the notebook
# File: notebooks/kaggle/AutoMixAI_Combined_Training.ipynb

# Run all cells
# (menu) Kernel → Restart & Run All
```

### Option 3: VS Code with Jupyter Extension

1. Install Jupyter extension in VS Code
2. Open the notebook file
3. Select kernel (Python 3.11)
4. Run cells sequentially or all at once

## Feature Extraction

Features extracted per audio frame:
- **Spectral Centroid**: Center of mass of spectrum (1 feature)
- **MFCC**: 13 mel-frequency cepstral coefficients
- **Zero Crossing Rate**: Average zero crossings per frame (1 feature)
- **Spectral Rolloff**: Frequency below which 85% of power lies (1 feature)
- **Chroma Features**: 12 pitch class estimates per frame

**Total: 28 features per frame**

## Model Performance Expectations

### Beat Detection
- **Accuracy**: 85-92% (depending on data quality)
- **Precision**: 80-90% (false positive avoidance)
- **Recall**: 70-85% (catching actual beats)
- **F1-Score**: 75-87% (balanced metric)

### Key Detection
- **Top-1 Accuracy**: 60-75% (exact key prediction)
- **Top-3 Accuracy**: 85-95% (within 3 nearest keys)

*Actual performance depends on data quality and quantity*

## Troubleshooting

### Issue: "Dataset not found"
```
Status: ✗ Ballroom not found
```
**Solution**: Verify datasets are in the correct paths:
- `Datasets/BallroomData/`
- `Datasets/fma_small/`
- `Datasets/medleydb/`
- `Datasets/giantsteps-key-dataset/`

### Issue: "ImportError: No module named 'librosa'"
**Solution**: Install missing packages
```bash
pip install librosa scipy soundfile pyyaml
```

### Issue: Out of memory
**Solution**: Reduce sample size in loader functions
```python
prepare_ballroom(max_files=50)  # Instead of full dataset
prepare_fma(max_files=50)
prepare_medleydb(max_files=20)
```

### Issue: CUDA out of memory (on GPU)
**Solution**: Reduce batch size
```python
batch_size=32  # Instead of 64
```

## Using Trained Models

### Load Models in Backend

```python
from tensorflow.keras.models import load_model
import joblib

# Load models and scalers
beat_model = load_model('beat_detection_model.h5')
key_model = load_model('key_detection_model.h5')
scaler_beat = joblib.load('scaler_beat.pkl')
scaler_key = joblib.load('scaler_key.pkl')

# Use on new audio
from librosa import load, feature, time_to_frames
y, sr = load('audio.wav')

# Extract features (same as training)
features = extract_features(y, sr)  # Returns (n_frames, n_features)

# Predict beats
features_scaled = scaler_beat.transform(features)
beat_probs = beat_model.predict(features_scaled)
beats = (beat_probs > 0.5).astype(int).flatten()

# Predict key (use frame average)
features_avg = features.mean(axis=0, keepdims=True)
features_avg_scaled = scaler_key.transform(features_avg)
key_probs = key_model.predict(features_avg_scaled)
key_pred = np.argmax(key_probs[0])
```

### Integrate with Backend Routes

```python
# In backend/app/routes/analyze.py
from app.model.inference import InferenceEngine

engine = InferenceEngine()
beat_data = engine.detect_beats(audio_path)
key_data = engine.detect_key(audio_path)

return {
    "beats": beat_data,
    "key": key_data,
    "confidence": scores
}
```

## Features & Capabilities

✅ **Multi-dataset training** - Combines 4 diverse datasets  
✅ **Automatic feature extraction** - MFCC, spectral, chroma  
✅ **Scalable architecture** - Works with partial datasets  
✅ **Production-ready models** - Can be deployed immediately  
✅ **Comprehensive metrics** - Accuracy, precision, recall, F1  
✅ **Model persistence** - Save/load for inference  
✅ **Kaggle submission ready** - Export in competition format  
✅ **Detailed reporting** - JSON summaries and markdown reports

## Next Steps After Training

1. **Test predictions** in `kaggle_submission.csv`
2. **Copy models** to `backend/app/storage/models/`
3. **Update backend** inference code to use new models
4. **Test in frontend** UI with sample uploads
5. **Performance monitoring** - Track metrics over time
6. **Model refinement** - Collect new data, retrain periodically

## Advanced Options

### Custom Feature Extraction
Modify `extract_features()` to include additional features:
```python
# Add tempogram (tempo-related features)
tempogram = librosa.feature.tempogram(y=y_audio, sr=sr)
features_list.append(tempogram.mean(axis=1))

# Add constant-Q transform
cqt = librosa.feature.chroma_cqt(y=y_audio, sr=sr)
features_list.extend(cqt)
```

### Ensemble Models
Train multiple models and average predictions:
```python
beat_pred = (model1.predict(X_test) + model2.predict(X_test)) / 2
beat_pred = (beat_pred > 0.5).astype(int)
```

### Transfer Learning
Use pre-trained models as starting point:
```python
from tensorflow.keras.applications import MobileNetV2
base_model = MobileNetV2(include_top=False, weights='imagenet')
# Add custom layers on top
```

## References

- **Librosa Documentation**: https://librosa.org/
- **TensorFlow/Keras**: https://www.tensorflow.org/
- **GiantstepsKey Dataset**: https://github.com/marl/giantsteps-key-dataset
- **MedleyDB**: https://github.com/marl/medleydb
- **Ballroom Dataset**: http://www.music-ir.org/mirex/abstracts/2012/ballroom.pdf
- **FMA Dataset**: https://github.com/mdeff/fma

---

**Last Updated**: March 15, 2026  
**Project**: AutoMixAI  
**Status**: ✅ Production Ready
