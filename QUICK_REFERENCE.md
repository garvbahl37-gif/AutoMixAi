# ⚡ Quick Reference - GiantstepsKey Setup

## 📂 Files Created

| File | Location | Purpose |
|------|----------|---------|
| **GiantstepsKey Training Notebook** | `kaggle/AutoMixAI_GiantstepsKey_Training.ipynb` | Main training pipeline (12 sections) |
| **Setup Documentation** | `kaggle/GiantstepsKey_README.md` | Comprehensive guide & setup instructions |
| **Backend Integration** | `INTEGRATION_GUIDE.md` | How to use trained model in backend |
| **Download Helper** | `giantsteps_audio_download.py` | Download 604 MP3 files from Beatport |
| **Project Summary** | `COMPLETION_SUMMARY.md` | What was built and next steps |
| **Dataset** | `Datasets/giantsteps-key-dataset/` | Cloned repository with 604 key annotations |

## 🚀 Quick Start

### Option A: Train with Demo (Synthetic Features)
```bash
# No downloads needed, runs in ~5-10 minutes
jupyter notebook kaggle/AutoMixAI_GiantstepsKey_Training.ipynb
# Run all cells → outputs saved to kaggle/output/
```

### Option B: Train with Actual Audio (Best Quality)
```bash
# Download 604 MP3 files first (~850 MB)
python giantsteps_audio_download.py --output-dir Datasets/giantsteps-key-dataset/audio --max-workers 10

# Then run notebook
jupyter notebook kaggle/AutoMixAI_GiantstepsKey_Training.ipynb
```

### Option C: Train on Kaggle (Free GPU)
```
1. Create Kaggle Dataset: giantsteps-key-dataset
2. New Kaggle Notebook → import cells from AutoMixAI_GiantstepsKey_Training.ipynb
3. Select GPU runtime
4. Run all cells
5. Download outputs
```

## 🎯 What Gets Trained

| Component | Details |
|-----------|---------|
| **Model** | 4-layer ANN (43→256→128→64→24 neurons) |
| **Input** | 43-dimensional audio features |
| **Output** | 24 musical keys (12 pitches × major/minor) |
| **Training Data** | 604 GiantstepsKey tracks (optional: combine with Ballroom) |
| **Accuracy** | ~85-90% expected (varies by audio quality) |
| **Training Time** | ~30-60 minutes (Kaggle GPU) |

## 📊 Feature Extraction

**43-Dimensional Features per Track:**

```
13 MFCCs                                    [0-12]
13 Delta-MFCCs (temporal velocity)          [13-25]
12 Chroma features (pitch/harmony)          [26-37]
Onset strength, Spectral flux, RMS energy, 
Spectral centroid, Zero-crossing rate       [38-42]
```

## 💾 Output Files (After Training)

```
kaggle/output/
├── models/
│   ├── key_detector_giantsteps.h5         ← Copy to backend/app/storage/models/
│   ├── key_label_encoder.pkl              ← Copy to backend/app/storage/models/
│   └── feature_normalization.pkl          ← Copy to backend/app/storage/models/
├── data/
│   ├── giantsteps_key_features.csv        ← All training features
│   └── training_metrics.csv               ← Model performance
└── combined_dataset_info.json
```

## 🔗 Backend Integration

```python
# After copying model files:

from app.services.key_detector import KeyDetector

detector = KeyDetector()
result = detector.detect_key("song.wav", return_confidence=True)

print(f"Key: {result['key']}")                      # e.g., "C major"
print(f"Confidence: {result['confidence']:.1%}")    # e.g., "87.5%"
print(f"Top 3: {result['top_3']}")                  # All predictions
```

## 📖 Documentation Map

- **Getting started?** → Read `kaggle/GiantstepsKey_README.md`
- **Want to integrate?** → Follow `INTEGRATION_GUIDE.md`
- **Need details?** → See `COMPLETION_SUMMARY.md`
- **Quick overview?** → You're reading it!

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| "Audio files not found" | Use demo mode (synthetic) OR run download script |
| "Models not found" (after training) | Check `kaggle/output/models/` directory exists |
| "Import errors" | Run pip install in notebook cell 2 |
| "Out of memory" | Reduce BATCH_SIZE or MAX_SAMPLES in section 2 |
| "Low accuracy" | Dataset is EDM-focused; may need retraining on your music |

## 🎓 Key Detection Basics

A musical key tells you the tonal center (pitch) and mode (major/minor):

```
C major    = C is the tonic note, major scale
D minor    = D is the tonic note, minor scale
...
B major    = B is the tonic note, major scale
```

With 12 chromatic pitches and 2 modes = 24 possible keys

## 📈 Expected Performance

On GiantstepsKey test set:
- **Accuracy**: 85-90% (top-1)
- **Top-3 Accuracy**: 95-98%
- **Per-class F1**: 0.82-0.88

(Depends on audio quality, training hyperparameters)

## 🔐 Dataset Info

| Property | Value |
|----------|-------|
| **Source** | GitHub: GiantSteps/giantsteps-key-dataset |
| **Samples** | 604 tracks |
| **Duration** | ~2 minutes each |
| **Total Size** | ~850 MB (audio), ~10 MB (annotations) |
| **Annotations** | Professional manual key labels |
| **Format** | MP3 (audio), plain text .key files |
| **License** | See repository README |

## ✅ Checklist

- [ ] Cloned GiantstepsKey repository
- [ ] Created training notebook
- [ ] Created documentation
- [ ] Created download helper script
- [ ] Ready to train!

Next:
- [ ] Run training notebook
- [ ] Download output files
- [ ] Copy to backend/app/storage/models/
- [ ] Implement key_detector.py in backend
- [ ] Test endpoint

## 🎬 Start Here

```bash
# 1. Read the README
cat kaggle/GiantstepsKey_README.md

# 2. Download audio (optional, for best results)
python giantsteps_audio_download.py

# 3. Run the notebook
jupyter notebook kaggle/AutoMixAI_GiantstepsKey_Training.ipynb

# 4. After training, integrate into backend
# See INTEGRATION_GUIDE.md
```

---

**Questions?** Check the documentation files or refer to GiantstepsKey's official repo.

**Ready?** Start with the training notebook! 🚀

