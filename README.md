<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">AutoMixAI</h1>

<p align="center">
  <strong>AI-Powered Automated DJ Mixing System</strong>
  <br/>
  <em>Beat Detection • Genre Classification • Instrument Recognition • Intelligent Mixing</em>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-datasets">Datasets</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-models">Models</a>
</p>

---

## Overview

AutoMixAI is a full-stack AI-powered music analysis and DJ mixing platform. Users upload audio tracks, and the system analyzes BPM, beats, genre, instruments, mood, and tags using neural networks. The platform can automatically mix two tracks with beat-aligned crossfades and generate custom drum patterns from natural language prompts.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AutoMixAI Platform                             │
├─────────────────────────────────────────────────────────────────────────────┤
│   Upload    →     Analyze    →     Mix    →    Generate           │
│  Audio Files    AI Analysis     DJ Mixing   Beat Synthesis        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### Audio Analysis
- **Beat Detection** - Neural network-based beat tracking with frame-level precision
- **BPM Estimation** - Accurate tempo detection using onset strength analysis
- **Genre Classification** - 10-class GTZAN classification with confidence scores
- **Instrument Recognition** - NSynth-trained 11-family instrument detection
- **Music Tagging** - Multi-label tagging with 56 descriptive tags
- **Mood Detection** - Automatic mood inference (energetic, calm, melancholic, intense)
- **Vocal Detection** - Identifies vocal presence vs instrumental tracks
- **Energy Analysis** - RMS-based energy level classification

### DJ Mixing
- **Beat Alignment** - Automatic beat-grid synchronization
- **Time Stretching** - Tempo matching without pitch distortion
- **Crossfade Engine** - Smooth DJ-style transitions
- **Multi-track Support** - Mix multiple tracks in sequence

### Beat Generation
- **Natural Language Prompts** - "Create a 120 BPM trap beat with heavy bass"
- **20+ Genre Patterns** - From hip-hop to ambient, trap to jazz
- **10 Drum Instruments** - Kick, snare, hi-hats, claps, toms, and more
- **Humanization** - Velocity and timing variations for natural feel
- **Time Signature Support** - 4/4, 3/4, 6/8 patterns

---

## Architecture

### System Overview

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                                   CLIENT LAYER                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                        React Frontend (Vite + ES6)                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐     │  │
│  │  │ Upload Page │  │Analyze Page │  │  Mix Page   │  │Beat Generator Page│     │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘     │  │
│  │         │                │                │                  │               │  │
│  │         └────────────────┴────────────────┴──────────────────┘               │  │
│  │                                   │                                           │  │
│  │                            API Client (fetch)                                 │  │
│  └───────────────────────────────────┼───────────────────────────────────────────┘  │
└──────────────────────────────────────┼──────────────────────────────────────────────┘
                                       │ HTTP/REST
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                   API LAYER                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │                         FastAPI Backend Server                                  │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐   │  │
│  │  │                              Routes                                      │   │  │
│  │  │   POST /upload    POST /analyze    POST /mix    POST /generate          │   │  │
│  │  │   GET /output/{id}                                                       │   │  │
│  │  └─────────────────────────────────┬───────────────────────────────────────┘   │  │
│  │                                    │                                            │  │
│  │  ┌─────────────────────────────────▼───────────────────────────────────────┐   │  │
│  │  │                            Schemas                                       │   │  │
│  │  │   AnalysisResponse • MixRequest • GenerateRequest • TagScore            │   │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬───────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                 SERVICE LAYER                                        │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │                           Audio Processing Services                             │  │
│  │                                                                                 │  │
│  │   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐              │  │
│  │   │  audio_loader   │   │ beat_detector   │   │ bpm_estimator   │              │  │
│  │   │  Load & decode  │   │  ANN inference  │   │  Tempo tracking │              │  │
│  │   └─────────────────┘   └─────────────────┘   └─────────────────┘              │  │
│  │                                                                                 │  │
│  │   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐              │  │
│  │   │genre_classifier │   │instrument_class │   │  tag_predictor  │              │  │
│  │   │  GTZAN 10-class │   │  NSynth 11-fam  │   │  MagnaTagATune  │              │  │
│  │   └─────────────────┘   └─────────────────┘   └─────────────────┘              │  │
│  │                                                                                 │  │
│  │   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐              │  │
│  │   │ drum_classifier │   │ beat_generator  │   │     mixer       │              │  │
│  │   │  Kick/Snare/HH  │   │  NLP + Synth    │   │  Time-stretch   │              │  │
│  │   └─────────────────┘   └─────────────────┘   └─────────────────┘              │  │
│  │                                                                                 │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬───────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                   ML LAYER                                           │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │                          Neural Network Models                                  │  │
│  │                                                                                 │  │
│  │   ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │   │  beat_detector.h5    │  Dense ANN (128→64→1) for beat activation        │  │  │
│  │   ├─────────────────────────────────────────────────────────────────────────┤  │  │
│  │   │  genre_classifier.h5 │  Dense (256→128→10) for GTZAN genres             │  │  │
│  │   ├─────────────────────────────────────────────────────────────────────────┤  │  │
│  │   │  nsynth_classifier.h5│  Dense (128→64→11) for instrument families       │  │  │
│  │   ├─────────────────────────────────────────────────────────────────────────┤  │  │
│  │   │  tag_predictor.h5    │  Dense (256→128→56) for multi-label tagging      │  │  │
│  │   ├─────────────────────────────────────────────────────────────────────────┤  │  │
│  │   │  drum_classifier.h5  │  Dense (64→32→4) for drum classification         │  │  │
│  │   └─────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                                 │  │
│  │   Feature Scalers: feature_scaler.pkl, genre_scaler.pkl, nsynth_scaler.pkl,   │  │
│  │                    tag_scaler.pkl, drum_scaler.pkl                             │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### Analysis Pipeline Flowchart

```
                              ┌─────────────────┐
                              │   Audio File    │
                              │   (WAV/MP3)     │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  audio_loader   │
                              │  Load @ 22050Hz │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │feature_extractor│
                              │  43-dim vector  │
                              └────────┬────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│   beat_detector     │   │  genre_classifier   │   │ instrument_classifier│
│                     │   │                     │   │                     │
│  Input: 43-dim      │   │  Input: 57-dim      │   │  Input: 43-dim      │
│  Output: P(beat)    │   │  Output: 10 classes │   │  Output: 11 families│
└──────────┬──────────┘   └──────────┬──────────┘   └──────────┬──────────┘
           │                         │                         │
           ▼                         ▼                         ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  bpm_estimator      │   │   Top-3 Genres      │   │  Dominant Instrument│
│  onset_strength     │   │   + Confidence %    │   │  + Top-3 w/ scores  │
└──────────┬──────────┘   └─────────────────────┘   └─────────────────────┘
           │
           ▼
┌─────────────────────┐          ┌─────────────────────┐
│   Beat Timestamps   │          │   tag_predictor     │
│   [0.45, 0.92, ...] │          │                     │
└─────────────────────┘          │  Input: 43-dim      │
                                 │  Output: 56 tags    │
                                 └──────────┬──────────┘
                                            │
                                 ┌──────────┴──────────┐
                                 ▼                     ▼
                      ┌─────────────────┐   ┌─────────────────┐
                      │  Mood Detection │   │  Vocal Detection│
                      │  (from tags)    │   │  (from tags)    │
                      └─────────────────┘   └─────────────────┘
```

### Mixing Engine Flowchart

```
     Track A                                           Track B
        │                                                 │
        ▼                                                 ▼
┌───────────────┐                                 ┌───────────────┐
│ Beat Detection│                                 │ Beat Detection│
│ BPM: 128      │                                 │ BPM: 125      │
│ Beats: [...]  │                                 │ Beats: [...]  │
└───────┬───────┘                                 └───────┬───────┘
        │                                                 │
        │              ┌─────────────────┐                │
        └──────────────▶  Target BPM     ◀────────────────┘
                       │  (126.5 avg)    │
                       └────────┬────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────┐                         ┌───────────────────┐
│   Time Stretch    │                         │   Time Stretch    │
│   128 → 126.5     │                         │   125 → 126.5     │
│   (pyrubberband)  │                         │   (pyrubberband)  │
└─────────┬─────────┘                         └─────────┬─────────┘
          │                                             │
          │           ┌───────────────────┐             │
          └───────────▶  Beat Alignment   ◀─────────────┘
                      │  Align on beat 1  │
                      └─────────┬─────────┘
                                │
                                ▼
                      ┌───────────────────┐
                      │  Crossfade Mix    │
                      │  ───────────────  │
                      │  A ████████░░░░░  │
                      │  B ░░░░░████████  │
                      │  ───────────────  │
                      │  Duration: 8 bars │
                      └─────────┬─────────┘
                                │
                                ▼
                      ┌───────────────────┐
                      │   Output WAV      │
                      │   Mixed Track     │
                      └───────────────────┘
```

### Beat Generator Flowchart

```
                    ┌────────────────────────────────────┐
                    │           User Prompt              │
                    │  "120 BPM trap beat, dark mood,   │
                    │   4 bars, heavy 808 bass"         │
                    └───────────────┬────────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────────┐
                    │         NLP Parser                 │
                    │  ┌──────────────────────────────┐  │
                    │  │ BPM: 120 (regex \d+ bpm)     │  │
                    │  │ Genre: trap (keyword match)  │  │
                    │  │ Mood: dark (mood keywords)   │  │
                    │  │ Bars: 4 (regex \d+ bar)      │  │
                    │  │ Energy: high (keyword match) │  │
                    │  └──────────────────────────────┘  │
                    └───────────────┬────────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────────┐
                    │       Pattern Generator            │
                    │                                    │
                    │  Genre: trap                       │
                    │  ┌────────────────────────────┐    │
                    │  │ kick:   [1,0,0,0,1,0,0,0,  │    │
                    │  │          1,0,1,0,0,0,0,0]  │    │
                    │  │ snare:  [0,0,0,0,1,0,0,0,  │    │
                    │  │          0,0,0,0,1,0,0,0]  │    │
                    │  │ hihat:  [1,1,1,1,1,1,1,1,  │    │
                    │  │          1,1,1,1,1,1,1,1]  │    │
                    │  └────────────────────────────┘    │
                    └───────────────┬────────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────────┐
                    │       Drum Synthesizer             │
                    │                                    │
                    │  ┌─────────┐  kick: sine + decay   │
                    │  │  Kick   │  60-150Hz sweep       │
                    │  └─────────┘                       │
                    │  ┌─────────┐  snare: noise + tone  │
                    │  │  Snare  │  200Hz + bandpass     │
                    │  └─────────┘                       │
                    │  ┌─────────┐  hihat: noise + HP    │
                    │  │  HiHat  │  8kHz highpass        │
                    │  └─────────┘                       │
                    └───────────────┬────────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────────┐
                    │       Humanization                 │
                    │                                    │
                    │  Velocity: ±15% random variation   │
                    │  Timing: ±5ms micro-shifts         │
                    │  Swing: 0-30% shuffle amount       │
                    └───────────────┬────────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────────┐
                    │       Output WAV                   │
                    │       44.1kHz, 16-bit              │
                    └────────────────────────────────────┘
```

---

## Datasets

AutoMixAI was trained on multiple diverse audio datasets to achieve robust performance across different music styles and analysis tasks.

### Training Datasets Summary

| Dataset | Task | Samples | Classes/Labels | Source |
|---------|------|---------|----------------|--------|
| **GTZAN** | Genre Classification | 1,000 | 10 genres | Kaggle |
| **NSynth** | Instrument Classification | 305,979 | 11 families | Kaggle |
| **MagnaTagATune** | Music Tagging | 25,863 | 56 tags | Kaggle |
| **Drum Kit Sounds** | Drum Classification | ~150 | 4 classes | Kaggle |
| **Lakh MIDI** | Pattern Generation | 45,000+ | Rhythm patterns | Kaggle |
| **Ballroom** | Beat Detection | 698 | Beat annotations | Research |
| **FMA Small** | Beat Detection | 8,000 | Beat annotations | Research |
| **MedleyDB** | Beat Detection | 122 | Beat annotations | Research |

### Detailed Dataset Information

#### GTZAN Genre Collection
**Purpose:** Genre classification model training

```
Source:     Kaggle - andradaolteanu/gtzan-dataset-music-genre-classification
Tracks:     1,000 (100 per genre × 10 genres)
Duration:   30 seconds each
Format:     WAV, 22050Hz mono
Genres:     blues, classical, country, disco, hiphop,
            jazz, metal, pop, reggae, rock
Features:   57-dimensional (chroma, spectral, MFCC)
Model:      Dense 256→128→10, softmax
Accuracy:   ~78% (10-class)
```

#### NSynth Music Dataset
**Purpose:** Instrument family classification

```
Source:     Kaggle - anubhavchhabra/nsynth-music-dataset
Samples:    305,979 musical notes
Duration:   4 seconds each
Format:     TFRecord (parsed to NumPy)
Families:   bass, brass, flute, guitar, keyboard, mallet,
            organ, reed, string, synth_lead, vocal
Features:   43-dimensional (MFCC, spectral, chroma)
Model:      Dense 128→64→11, softmax
```

#### MagnaTagATune Dataset
**Purpose:** Multi-label music tagging, mood detection, vocal detection

```
Source:     Kaggle - shrirangmahajan/magnatagatune
Clips:      25,863 audio clips (after filtering)
Duration:   ~30 seconds each
Format:     MP3
Labels:     188 original → 56 valid tags (filtered for quality)
Tags:       guitar, piano, drums, female voice, fast, slow,
            rock, electronic, classical, ambient, etc.
Features:   43-dimensional
Model:      Dense 256→128→56, sigmoid (multi-label)
Threshold:  0.3 for tag activation
```

#### Drum Kit Sound Samples
**Purpose:** Drum hit classification for pattern analysis

```
Source:     Kaggle - sparshgupta/drum-kit-sound-samples
Samples:    ~150 isolated drum hits
Classes:    kick, snare, hihat, tom
Format:     WAV
Features:   43-dimensional (onset + spectral focused)
Model:      Dense 64→32→4, softmax
Note:       Studio-quality samples, may need augmentation
            for real-world generalization
```

#### Lakh MIDI Dataset
**Purpose:** Rhythm pattern extraction for beat generation

```
Source:     Kaggle - federicodellellis/lakh-midi-dataset-clean
Files:      45,000+ MIDI files
Content:    Full songs with drum tracks
Extracted:  Drum onset patterns, velocity information
Usage:      Pattern templates for beat generator
Library:    pretty_midi for parsing
Output:     midi_patterns.pkl (quantized patterns)
```

#### Ballroom Dataset
**Purpose:** Beat detection model training

```
Source:     Research dataset (ballroomdancers.com)
Tracks:     698 dance music excerpts
Duration:   ~30 seconds each
Genres:     Waltz, Tango, Foxtrot, Quickstep,
            Viennese Waltz, Samba, ChaCha, Rumba, Jive
Labels:     Beat timestamp annotations (CSV)
Features:   43-dimensional per frame
```

#### FMA Small
**Purpose:** Beat detection model training (diverse genres)

```
Source:     Free Music Archive
Tracks:     8,000
Duration:   30 seconds each
Genres:     8 balanced genres
Labels:     Beat annotations via librosa
Features:   43-dimensional per frame
```

#### MedleyDB
**Purpose:** Beat detection for professional multitrack recordings

```
Source:     NYU Music and Audio Research Lab
Tracks:     122 multitrack songs
Content:    Full studio productions
Labels:     Expert beat annotations
Usage:      Fine-tuning beat detector on complex mixes
```

### Feature Extraction

All audio features are extracted using librosa with consistent parameters:

```python
# Common Parameters
SAMPLE_RATE = 22050
HOP_LENGTH = 1024  # ~46ms per frame
N_FFT = 2048
N_MELS = 128
N_MFCC = 20

# 43-Dimensional Feature Vector (per frame)
features = [
    mfcc[0:13],           # 13 dims - Timbre
    spectral_centroid,    # 1 dim  - Brightness
    spectral_bandwidth,   # 1 dim  - Spread
    spectral_rolloff,     # 1 dim  - High-freq energy
    spectral_contrast[7], # 7 dims - Harmonic structure
    spectral_flatness,    # 1 dim  - Noise vs tone
    zero_crossing_rate,   # 1 dim  - Percussive content
    rms_energy,           # 1 dim  - Loudness
    onset_strength,       # 1 dim  - Transient detection
    chroma[0:12],         # 12 dims - Pitch class
    tempo_feature,        # 1 dim  - BPM context
    beat_sync,            # 1 dim  - Beat alignment
    delta_mfcc[0:1],      # 1 dim  - Temporal change
]
```

---

## Project Structure

```
AutoMixAI/
│
├── backend/                          # FastAPI Backend
│   └── app/
│       ├── main.py                   # Application entry point
│       │
│       ├── routes/                   # API Endpoints
│       │   ├── upload.py             # POST /upload
│       │   ├── analyze.py            # POST /analyze
│       │   ├── mix.py                # POST /mix
│       │   └── generate.py           # POST /generate
│       │
│       ├── schemas/                  # Pydantic Models
│       │   ├── analysis_response.py  # AnalysisResponse, TagScore
│       │   └── generate_request.py   # GenerateRequest
│       │
│       ├── services/                 # Core Services
│       │   ├── audio_loader.py       # Audio file loading
│       │   ├── beat_detector.py      # ANN beat detection
│       │   ├── bpm_estimator.py      # Tempo estimation
│       │   ├── feature_extractor.py  # 43-dim feature extraction
│       │   ├── genre_classifier.py   # GTZAN classification
│       │   ├── instrument_classifier.py  # NSynth classification
│       │   ├── tag_predictor.py      # MagnaTagATune tagging
│       │   ├── drum_classifier.py    # Drum hit classification
│       │   ├── beat_generator.py     # NLP-based beat synthesis
│       │   └── mixer.py              # DJ mixing engine
│       │
│       ├── model/                    # ML Components
│       │   ├── ann_model.py          # Model architecture
│       │   ├── train.py              # Training script
│       │   └── inference.py          # Model inference
│       │
│       ├── data/                     # Data Processing
│       │   ├── prepare_data.py       # Dataset → X.npy/y.npy
│       │   └── medleydb_loader.py    # MedleyDB loader
│       │
│       ├── utils/                    # Utilities
│       │   ├── config.py             # Settings (pydantic-settings)
│       │   ├── logger.py             # Logging configuration
│       │   └── helpers.py            # Helper functions
│       │
│       └── storage/                  # File Storage
│           ├── uploads/              # Uploaded audio files
│           ├── output/               # Generated mixes
│           └── models/               # Trained .h5 models
│               ├── beat_detector.h5
│               ├── feature_scaler.pkl
│               ├── genre_classifier.h5
│               ├── genre_scaler.pkl
│               ├── nsynth_classifier.h5
│               ├── nsynth_scaler.pkl
│               ├── nsynth_labels.pkl
│               ├── tag_predictor.h5
│               ├── tag_scaler.pkl
│               ├── tag_labels.pkl
│               ├── drum_classifier.h5
│               ├── drum_scaler.pkl
│               └── midi_patterns.pkl
│
├── frontend/                         # React Frontend
│   ├── index.html                    # HTML entry point
│   ├── vite.config.js                # Vite configuration
│   ├── package.json                  # Dependencies
│   └── src/
│       ├── main.jsx                  # React entry point
│       ├── App.jsx                   # Main app + routing
│       ├── api.js                    # API client
│       ├── index.css                 # Global styles
│       ├── components/               # Shared components
│       │   ├── WaveformPlayer.jsx
│       │   ├── StatCard.jsx
│       │   └── GenreBar.jsx
│       └── pages/                    # Page components
│           ├── UploadPage.jsx
│           ├── AnalyzePage.jsx
│           ├── MixPage.jsx
│           └── BeatGeneratorPage.jsx
│
├── notebooks/                        # Jupyter Notebooks
│   └── kaggle/
│       ├── GenreClassifier_Training.ipynb
│       └── MultiTask_Audio_Training.ipynb
│
├── Datasets/                         # Local training data
│   ├── BallroomAnnotations/
│   ├── BallroomData/
│   ├── fma_small/
│   └── medleydb/
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- FFmpeg (for audio processing)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/AutoMixAI.git
cd AutoMixAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Download models (if not training from scratch)
# Place .h5 and .pkl files in backend/app/storage/models/
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Running the Application

```bash
# Terminal 1: Start backend
cd backend
uvicorn app.main:app --reload --port 8002

# Terminal 2: Start frontend
cd frontend
npm run dev
```

Access the application at `http://localhost:5173`

---

## 📡 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload audio file |
| `POST` | `/analyze` | Analyze uploaded track |
| `POST` | `/mix` | Mix two tracks |
| `POST` | `/generate` | Generate beat from prompt |
| `GET` | `/output/{file_id}` | Download output file |

### POST /upload

Upload an audio file for analysis.

**Request:**
```bash
curl -X POST "http://localhost:8002/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@track.mp3"
```

**Response:**
```json
{
  "file_id": "abc123-def456-ghi789",
  "filename": "track.mp3"
}
```

### POST /analyze

Analyze an uploaded track.

**Request:**
```json
{
  "file_id": "abc123-def456-ghi789"
}
```

**Response:**
```json
{
  "bpm": 128.5,
  "beat_times": [0.45, 0.92, 1.38, 1.85, ...],
  "duration": 180.5,
  "genre": "hiphop",
  "genre_confidence": 0.87,
  "genre_top3": [
    {"genre": "hiphop", "confidence": 0.87},
    {"genre": "pop", "confidence": 0.08},
    {"genre": "jazz", "confidence": 0.03}
  ],
  "dominant_instrument": "keyboard",
  "instrument_confidence": 0.72,
  "instruments_top3": [
    {"instrument": "keyboard", "confidence": 0.72},
    {"instrument": "bass", "confidence": 0.15},
    {"instrument": "synth_lead", "confidence": 0.08}
  ],
  "tags": ["piano", "drums", "slow", "ambient"],
  "tag_scores": [
    {"tag": "piano", "score": 0.89},
    {"tag": "drums", "score": 0.76}
  ],
  "mood": "calm",
  "has_vocals": false,
  "energy": "medium"
}
```

### POST /mix

Mix two uploaded tracks.

**Request:**
```json
{
  "track_a_id": "file-id-1",
  "track_b_id": "file-id-2",
  "crossfade_duration": 8.0
}
```

**Response:**
```json
{
  "output_file_id": "mix-xyz789",
  "duration": 240.5,
  "target_bpm": 126.5
}
```

### POST /generate

Generate a drum beat from a natural language prompt.

**Request:**
```json
{
  "prompt": "Create a 120 BPM trap beat with heavy 808s, 4 bars, dark mood"
}
```

**Response:**
```json
{
  "output_file_id": "beat-abc123",
  "bpm": 120,
  "bars": 4,
  "genre": "trap",
  "pattern": {
    "kick": [1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0],
    "snare": [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
    "hihat": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
  }
}
```

---

## Models

### Model Architectures

#### Beat Detector
```
Input: 43-dim feature vector
Dense(128, relu) → Dropout(0.3)
Dense(64, relu) → Dropout(0.3)
Dense(1, sigmoid)
Output: P(beat) ∈ [0, 1]
```

#### Genre Classifier
```
Input: 57-dim feature vector
Dense(256, relu) → BatchNorm → Dropout(0.4)
Dense(128, relu) → BatchNorm → Dropout(0.4)
Dense(10, softmax)
Output: 10-class probabilities
```

#### Instrument Classifier
```
Input: 43-dim feature vector
Dense(128, relu) → Dropout(0.3)
Dense(64, relu) → Dropout(0.3)
Dense(11, softmax)
Output: 11-family probabilities
```

#### Tag Predictor
```
Input: 43-dim feature vector
Dense(256, relu) → Dropout(0.4)
Dense(128, relu) → Dropout(0.4)
Dense(56, sigmoid)
Output: Multi-label scores (threshold: 0.3)
```

### Training

To train models from scratch:

```bash
# Prepare beat detection data
python -m backend.app.data.prepare_data

# Train beat detector
python -m backend.app.model.train

# For other models, use Kaggle notebooks:
# notebooks/kaggle/GenreClassifier_Training.ipynb
# notebooks/kaggle/MultiTask_Audio_Training.ipynb
```

---

## Configuration

Environment variables (prefix: `AUTOMIX_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTOMIX_SAMPLE_RATE` | 22050 | Audio sample rate |
| `AUTOMIX_HOP_LENGTH` | 512 | STFT hop length (training) |
| `AUTOMIX_MODEL_PATH` | `storage/models/beat_detector.h5` | Beat model path |
| `AUTOMIX_GENRE_MODEL_PATH` | `storage/models/genre_classifier.h5` | Genre model |
| `AUTOMIX_DEBUG` | false | Enable debug logging |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18, Vite, WaveSurfer.js, Lucide Icons |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **ML/AI** | TensorFlow/Keras, scikit-learn, NumPy |
| **Audio** | librosa, soundfile, pyrubberband |
| **Storage** | File-based (uploads/, output/) |

---

## Performance

| Task | Metric | Value |
|------|--------|-------|
| Beat Detection | F1-score | ~0.85 |
| BPM Estimation | Accuracy (±2 BPM) | ~92% |
| Genre Classification | Accuracy (10-class) | ~78% |
| Instrument Classification | Accuracy (11-class) | ~71% |
| Tag Prediction | mAP@10 | ~0.68 |

---

## Roadmap

- [ ] Real-time streaming analysis
- [ ] Key detection for harmonic mixing
- [ ] Energy-based track matching
- [ ] Playlist-level intelligent mixing
- [ ] Reinforcement learning DJ transitions
- [ ] Mobile app (React Native)
- [ ] Cloud deployment (AWS/GCP)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [librosa](https://librosa.org/) - Audio analysis library
- [TensorFlow](https://tensorflow.org/) - Machine learning framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [GTZAN Dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) - Genre classification
- [NSynth Dataset](https://magenta.tensorflow.org/nsynth) - Instrument sounds
- [MagnaTagATune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) - Music tagging

---

<p align="center">
  <strong>Built with 🎧 by AutoMixAI Team</strong>
</p>
