# GiantstepsKey Dataset Training Guide

## Overview

This guide provides instructions for training an **Artificial Neural Network (ANN)** on the **GiantstepsKey dataset** for musical key detection. The dataset contains 604 audio previews from Beatport with professional musical key annotations.

## What is GiantstepsKey?

- **604 audio tracks** (2-minute previews from Beatport)
- **Professional annotations** for musical key (e.g., "C major", "D# minor")
- **Electronic dance music** (EDM) and other genres
- Freely available via GitHub: https://github.com/GiantSteps/giantsteps-key-dataset

## Project Structure

```
AutoMixAI/
├── kaggle/
│   ├── AutoMixAI_GiantstepsKey_Training.ipynb   ← Training notebook
│   ├── AutoMixAI_Training_v2.ipynb              ← Beat detection (ballroom)
│   └── GiantstepsKey_README.md                  ← This file
└── Datasets/
    ├── giantsteps-key-dataset/                  ← Cloned repository
    │   ├── annotations/
    │   │   ├── key/                             ← Key labels (*.key files)
    │   │   └── ...
    │   └── audio/                               ← Audio files (if downloaded)
    └── BallroomAnnotations/                     ← Existing ballroom data
```

## Training Notebook Structure

The notebook `AutoMixAI_GiantstepsKey_Training.ipynb` contains:

1. **Dependencies Installation** - Installs required libraries
2. **Configuration** - Sets up paths, audio parameters, training hyperparameters
3. **Dataset Cloning** - Clones GiantstepsKey repo and explores annotations
4. **Feature Extraction** - Extracts 43-dimensional audio features per track
5. **Data Processing** - Loads and encodes key labels for training
6. **CSV Export** - Creates Kaggle-compatible CSV with features and labels
7. **Label Encoding** - Converts key names to numerical classes
8. **ANN Model Building** - Constructs neural network (256→128→64→24 neurons)
9. **Model Training** - Trains with early stopping, checkpoints, class weights
10. **Evaluation** - Tests model accuracy and generates metrics
11. **Model Serialization** - Saves model, encoder, and normalization state
12. **Optional Ballroom Combination** - Merges with ballroom dataset if available
13. **Results Summary** - Exports all outputs

## Audio Feature Matrix (43-dimensional)

The model uses rich audio features per track:

```
[0-12]   MFCC coefficients (Mel-Frequency Cepstral Coefficients)
[13-25]  Delta-MFCCs (temporal velocity of MFCCs)
[26-37]  Chroma STFT (12-bin chromatic pitch content)
[38]     Onset strength (mean)
[39]     Spectral flux (mean)
[40]     RMS energy (mean)
[41]     Spectral centroid (brightness, mean)
[42]     Zero-crossing rate (noisiness, mean)
```

These features capture:
- Timbral characteristics (MFCCs)
- Temporal rhythm patterns (delta-MFCCs, onset, flux)
- Harmonic/key-related content (chroma)
- Overall loudness and spectral characteristics

## ANN Model Architecture

```
Input: 43-dim audio features
  ↓
Dense(256) + BatchNorm + ReLU + Dropout(0.3)
  ↓
Dense(128) + BatchNorm + ReLU + Dropout(0.3)
  ↓
Dense(64) + BatchNorm + ReLU + Dropout(0.2)
  ↓
Output: Dense(24, softmax)  [12 pitches × 2 modes: major/minor]
```

**Key Design Choices:**
- **BatchNormalization** - Stabilizes training
- **Dropout** - Prevents overfitting
- **Class weighting** - Handles imbalanced key distribution
- **Early stopping** - Stops when validation loss plateaus

## Setup Instructions

### 1. Local Setup (with actual audio)

```bash
# Install dependencies
pip install librosa soundfile tensorflow numpy scipy pandas scikit-learn

# Optional: Download audio files
cd Datasets/giantsteps-key-dataset
bash audio_dl.sh                    # Downloads 604 MP3 files (~850 MB)
bash convert_audio.sh               # Converts MP3 → WAV (requires sox)

# Run Jupyter notebook
jupyter notebook kaggle/AutoMixAI_GiantstepsKey_Training.ipynb
```

### 2. Kaggle Notebook Setup

Upload to Kaggle:
1. Create a new Dataset: `giantsteps-key-dataset`
2. Upload the cloned repo contents OR just the annotations
3. Create a new Notebook and copy cells from `AutoMixAI_GiantstepsKey_Training.ipynb`
4. Update paths in Section 2 to use `/kaggle/input/` directories

### 3. Without Audio Files (Demo Mode)

The notebook will:
1. Clone the dataset (gets annotations only)
2. **Generate synthetic features** for demonstration
3. Still train and evaluate the ANN model
4. Export training data and metrics

This mode is useful for:
- Testing the pipeline
- Understanding feature extraction
- Validating model training without large downloads

## Expected Output Files

After training, outputs are saved to `./output/`:

```
output/
├── models/
│   ├── key_detector_giantsteps.h5          ← Trained model
│   ├── key_label_encoder.pkl               ← Label encoder (for inference)
│   └── feature_normalization.pkl           ← Feature scaling params
├── data/
│   ├── giantsteps_key_features.csv         ← Feature matrix (for Kaggle/ML)
│   └── training_metrics.csv                ← Accuracy, loss, timing
└── combined_dataset_info.json              ← Metadata about datasets
```

## Key Annotations Format

Each key file contains a single line:

```
C major
D minor
E# major
...
```

The dataset includes **24 possible keys** (12 pitches × 2 modes):

```
Pitches:  C, C#, D, D#, E, F, F#, G, G#, A, A#, B
Modes:    major, minor
```

## Using the Trained Model for Inference

```python
import tensorflow as tf
import numpy as np
import pickle

# Load model
model = tf.keras.models.load_model('output/models/key_detector_giantsteps.h5')

# Load encoder and normalizer
with open('output/models/key_label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('output/models/feature_normalization.pkl', 'rb') as f:
    norm_params = pickle.load(f)

# Inference on new audio
# 1. Extract 43-dim features from audio file
features = extract_features(y, sr)  # Shape: (43,)

# 2. Normalize
features_norm = (features - norm_params['mean']) / norm_params['std']

# 3. Predict
prediction = model.predict(features_norm.reshape(1, -1))
key_idx = np.argmax(prediction[0])
predicted_key = encoder.classes_[key_idx]
confidence = prediction[0][key_idx]

print(f"Predicted key: {predicted_key} ({confidence:.2%})")
```

## Training Hyperparameters

In the notebook (Section 2), adjust:

```python
EPOCHS = 100              # Number of training iterations
BATCH_SIZE = 32           # Samples per gradient update
PATIENCE = 15             # Early stopping patience
VALIDATION_SPLIT = 0.2    # Fraction for validation
TEST_SPLIT = 0.1          # Fraction for testing
```

## Tips for Improvement

1. **More data**: Combine with other key detection datasets (e.g., Ballroom, FMA)
2. **Data augmentation**: Pitch shift, time stretch, noise injection
3. **Ensemble**: Train multiple models with different random seeds
4. **Larger model**: Increase Dense layer sizes (256→512, etc.)
5. **Different architectures**: Try CNN or RNN models
6. **Multi-task learning**: Jointly learn key + style/genre

## Troubleshooting

### Issue: "Audio files not found"
**Solution**: Download using `audio_dl.sh` OR use demo mode (synthetic features)

### Issue: "Out of memory"
**Solution**: Reduce BATCH_SIZE or MAX_SAMPLES in Section 2

### Issue: "Model not converging (loss plateauing)"
**Solution**: 
- Reduce learning rate
- Increase regularization (more dropout)
- Check feature scaling

### Issue: "Unbalanced key distribution (class imbalance)"
**Solution**: Model already uses `class_weight` to handle this automatically

## References

- **GiantstepsKey Dataset**: https://github.com/GiantSteps/giantsteps-key-dataset
- **ISMIR 2015 Paper**: Knees et al., "Two data sets for tempo estimation and key detection in electronic dance music..."
- **Librosa Documentation**: https://librosa.org/doc/latest/
- **TensorFlow/Keras**: https://www.tensorflow.org/api_docs

## Integration with AutoMixAI Backend

To use this model in the backend:

```bash
# Copy model files
cp output/models/key_detector_giantsteps.h5 
   backend/app/storage/models/

cp output/models/key_label_encoder.pkl 
   backend/app/storage/models/

cp output/models/feature_normalization.pkl 
   backend/app/storage/models/
```

Then update inference code in `backend/app/routes/analyze.py` or similar.

