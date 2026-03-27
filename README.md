# AutoMixAI
**AI-Powered Automated DJ Mixing System**
> Beat Detection • Genre Classification • Instrument Recognition • Intelligent Mixing • Neural Beat Generation

[Features](#features) • [Architecture](#architecture) • [Datasets](#datasets) • [Installation](#installation) • [API](#api-reference) • [Models](#models)

---

## Overview

AutoMixAI is a full-stack AI-powered music analysis and DJ mixing platform. Users upload audio tracks, and the system analyzes BPM, beats, genre, instruments, mood, and tags using neural networks. The platform can automatically mix two tracks with beat-aligned crossfades and generate full, production-ready beats from natural language prompts using a fine-tuned generative audio model.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AutoMixAI Platform                             │
├─────────────────────────────────────────────────────────────────────────────┤
│   Upload    →     Analyze    →     Mix    →    Generate                     │
│  Audio Files    AI Analysis     DJ Mixing   Neural Beat Synthesis           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### Audio Analysis
- **Beat Detection** — Neural network-based beat tracking with frame-level precision
- **BPM Estimation** — Accurate tempo detection using onset strength analysis
- **Genre Classification** — 10-class GTZAN classification with confidence scores
- **Instrument Recognition** — NSynth-trained 11-family instrument detection
- **Music Tagging** — Multi-label tagging with 56 descriptive tags
- **Mood Detection** — Automatic mood inference (energetic, calm, melancholic, intense)
- **Vocal Detection** — Identifies vocal presence vs instrumental tracks
- **Energy Analysis** — RMS-based energy level classification
- **Key & Scale Detection** — Essentia-based key/scale extraction (confidence > 70%) for harmonic mixing

### DJ Mixing
- **Beat Alignment** — Automatic beat-grid synchronization
- **Time Stretching** — Tempo matching without pitch distortion
- **Crossfade Engine** — Smooth DJ-style transitions
- **Multi-track Support** — Mix multiple tracks in sequence

### Beat Generation
AutoMixAI's beat generator is powered by **StableBeaT** — a fine-tuned version of [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) trained on 20,000 trap/rap/R&B instrumentals. It moves beyond pattern-based synthesis to produce full, nuanced, production-ready audio from natural language prompts.

- **Natural Language Prompts** — *"Create a 120 BPM trap beat with heavy 808 bass, dark mood, 4 bars"*
- **Rich Subgenre Coverage** — Cloud trap, lo-fi jazz rap, chillhop, neo-soul, EDM, industrial hip-hop, and more
- **Instrumentation Awareness** — Synth bells, plucked bass, deep sub, Rhodes keys, vocal adlibs, 808s, and beyond
- **Key & Mood Conditioning** — Generate in a specific key/scale with targeted emotional character
- **Humanization** — Velocity and timing variations for natural, non-quantized feel
- **Inference Speed** — ~1 min 15 sec per generation on RTX 4050; 200 steps, CFG scale 7

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
│  │   │  Kick/Snare/HH  │   │ StableBeaT SAO  │   │  Time-stretch   │              │  │
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
│  │   │  beat_detector.h5      │  Dense ANN (128→64→1) for beat activation      │  │  │
│  │   ├─────────────────────────────────────────────────────────────────────────┤  │  │
│  │   │  genre_classifier.h5   │  Dense (256→128→10) for GTZAN genres           │  │  │
│  │   ├─────────────────────────────────────────────────────────────────────────┤  │  │
│  │   │  nsynth_classifier.h5  │  Dense (128→64→11) for instrument families     │  │  │
│  │   ├─────────────────────────────────────────────────────────────────────────┤  │  │
│  │   │  tag_predictor.h5      │  Dense (256→128→56) for multi-label tagging    │  │  │
│  │   ├─────────────────────────────────────────────────────────────────────────┤  │  │
│  │   │  drum_classifier.h5    │  Dense (64→32→4) for drum classification       │  │  │
│  │   ├─────────────────────────────────────────────────────────────────────────┤  │  │
│  │   │  stable_beat/ (LoRA)   │  SAO 1.0 fine-tuned on 40k trap/rap segments   │  │  │
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

The beat generator uses **StableBeaT**, a LoRA fine-tune of Stable Audio Open 1.0, paired with an NLP preprocessing pipeline and Llama-assisted prompt enrichment.

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
                    │  │ BPM:   120 (regex \d+ bpm)   │  │
                    │  │ Genre: trap (keyword match)  │  │
                    │  │ Mood:  dark (mood keywords)  │  │
                    │  │ Bars:  4 (regex \d+ bar)     │  │
                    │  │ Key:   inferred or explicit  │  │
                    │  │ Energy: high (keyword match) │  │
                    │  └──────────────────────────────┘  │
                    └───────────────┬────────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────────┐
                    │   Llama 3.1 3B Prompt Enrichment   │
                    │                                    │
                    │  Expands parsed params into a      │
                    │  rich, human-readable prompt with  │
                    │  synonym variety and subgenre      │
                    │  nuance for the generative model   │
                    │                                    │
                    │  Input:  {genre, bpm, mood, key,   │
                    │           instruments, energy}     │
                    │  Output: "A dark trap beat at      │
                    │  120 BPM in C minor, featuring     │
                    │  heavy 808 bass and synth bells..." │
                    └───────────────┬────────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────────┐
                    │     StableBeaT (SAO LoRA)          │
                    │                                    │
                    │  Fine-tuned Stable Audio Open 1.0  │
                    │  • 40k trap/rap/R&B segments       │
                    │  • ~277h of audio, 14 epochs       │
                    │  • Trained on A100 (~42h)          │
                    │  • 200 steps, CFG scale 7          │
                    │  • Duration: up to 47s per output  │
                    │                                    │
                    │  Learns: synth bells, deep sub,    │
                    │  plucked bass, 808s, vocal chops,  │
                    │  trap rhythms, harmonic structures │
                    └───────────────┬────────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────────┐
                    │       Output WAV                   │
                    │       44.1kHz, 16-bit              │
                    │       Full generative beat audio   │
                    └────────────────────────────────────┘
```

---

## Datasets

AutoMixAI was trained on multiple diverse audio datasets to achieve robust performance across different music styles and analysis tasks.

### Training Datasets Summary

| Dataset | Task | Samples | Classes/Labels | Source |
|---|---|---|---|---|
| GTZAN | Genre Classification | 1,000 | 10 genres | Kaggle |
| NSynth | Instrument Classification | 305,979 | 11 families | Kaggle |
| MagnaTagATune | Music Tagging | 25,863 | 56 tags | Kaggle |
| Drum Kit Sounds | Drum Classification | ~150 | 4 classes | Kaggle |
| Lakh MIDI | Pattern Generation | 45,000+ | Rhythm patterns | Kaggle |
| Trap/Rap Beats | Beat Generation (generative) | 40,000 segments (20k tracks) | Instruments, mood, genre, key, BPM | Custom |
| Ballroom | Beat Detection | 698 | Beat annotations | Research |
| FMA Small | Beat Detection | 8,000 | Beat annotations | Research |
| MedleyDB | Beat Detection | 122 | Beat annotations | Research |

---

### Detailed Dataset Information

#### GTZAN Genre Collection
```
Purpose:  Genre classification model training
Source:   Kaggle - andradaolteanu/gtzan-dataset-music-genre-classification
Tracks:   1,000 (100 per genre × 10 genres)
Duration: 30 seconds each
Format:   WAV, 22050Hz mono
Genres:   blues, classical, country, disco, hiphop,
          jazz, metal, pop, reggae, rock
Features: 57-dimensional (chroma, spectral, MFCC)
Model:    Dense 256→128→10, softmax
Accuracy: ~78% (10-class)
```

#### NSynth Music Dataset
```
Purpose:  Instrument family classification
Source:   Kaggle - anubhavchhabra/nsynth-music-dataset
Samples:  305,979 musical notes
Duration: 4 seconds each
Format:   TFRecord (parsed to NumPy)
Families: bass, brass, flute, guitar, keyboard, mallet,
          organ, reed, string, synth_lead, vocal
Features: 43-dimensional (MFCC, spectral, chroma)
Model:    Dense 128→64→11, softmax
```

#### MagnaTagATune Dataset
```
Purpose:  Multi-label music tagging, mood detection, vocal detection
Source:   Kaggle - shrirangmahajan/magnatagatune
Clips:    25,863 audio clips (after filtering)
Duration: ~30 seconds each
Format:   MP3
Labels:   188 original → 56 valid tags (filtered for quality)
Tags:     guitar, piano, drums, female voice, fast, slow,
          rock, electronic, classical, ambient, etc.
Features: 43-dimensional
Model:    Dense 256→128→56, sigmoid (multi-label)
Threshold: 0.3 for tag activation
```

#### Drum Kit Sound Samples
```
Purpose:  Drum hit classification for pattern analysis
Source:   Kaggle - sparshgupta/drum-kit-sound-samples
Samples:  ~150 isolated drum hits
Classes:  kick, snare, hihat, tom
Format:   WAV
Features: 43-dimensional (onset + spectral focused)
Model:    Dense 64→32→4, softmax
Note:     Studio-quality samples, may need augmentation
          for real-world generalization
```

#### Lakh MIDI Dataset
```
Purpose:  Rhythm pattern extraction for beat generation templates
Source:   Kaggle - federicodellellis/lakh-midi-dataset-clean
Files:    45,000+ MIDI files
Content:  Full songs with drum tracks
Extracted: Drum onset patterns, velocity information
Usage:    Pattern templates for beat generator
Library:  pretty_midi for parsing
Output:   midi_patterns.pkl (quantized patterns)
```

#### Trap/Rap Beats Dataset (StableBeaT Training)
```
Purpose:  Fine-tuning Stable Audio Open 1.0 for modern beat generation
Tracks:   20,000 trap/rap/R&B instrumentals
Subgenres: Cloud trap, lo-fi jazz rap, R&B, EDM, industrial hip-hop,
           jazzy chillhop, neo-soul, boom bap
Segments: 2 × 20–35s per track → 40,000 total audio segments
Duration: ~277 hours of audio
Tagging:  CLAP LAION model — instruments, moods, genres per segment
BPM/Key:  Essentia deeptemp-k16-3 (confidence > 70%)
Prompt Gen: Llama 3.1 3B (local) for human-readable natural language prompts
```

Each segment is annotated with rich metadata:

```json
{
  "39118.wav": {
    "instruments_tags": ["plucked guitar", "synth bells", "movie sample"],
    "genres_tags": ["rap with soul"],
    "moods_tags": ["trap melancholic", "love"],
    "key": "G",
    "scale": "minor",
    "tempo": 109.0,
    "start": 63,
    "duration": 26
  }
}
```

Final training prompts generated by Llama 3.1 3B:
```json
{
  "filepath": "39118.wav",
  "start": 63,
  "duration": 26,
  "prompt": "A melancholic and love-inspired rap with soul beat at 109 BPM in G minor, using plucked guitar, synth bells, and movie sample."
}
```

**Tag Embedding Clusters (T5-Base)**

T5-Base encodes the dataset tags into five semantically distinct groups:
- **Emotion** — cheerful, joyful, dreamy
- **Groove** — swing groove, nylon guitar, movie sample
- **Genre** — g-funk, chill rap beat, jazzy chillhop
- **Sonority** — trap vocal, trap guitar

Silhouette Score: **0.095** — clusters are intentionally close, reflecting the semantic density of trap music's vocabulary.

#### Ballroom Dataset
```
Purpose:  Beat detection model training
Source:   Research dataset (ballroomdancers.com)
Tracks:   698 dance music excerpts
Duration: ~30 seconds each
Genres:   Waltz, Tango, Foxtrot, Quickstep,
          Viennese Waltz, Samba, ChaCha, Rumba, Jive
Labels:   Beat timestamp annotations (CSV)
Features: 43-dimensional per frame
```

#### FMA Small
```
Purpose:  Beat detection model training (diverse genres)
Source:   Free Music Archive
Tracks:   8,000
Duration: 30 seconds each
Genres:   8 balanced genres
Labels:   Beat annotations via librosa
Features: 43-dimensional per frame
```

#### MedleyDB
```
Purpose:  Beat detection for professional multitrack recordings
Source:   NYU Music and Audio Research Lab
Tracks:   122 multitrack songs
Content:  Full studio productions
Labels:   Expert beat annotations
Usage:    Fine-tuning beat detector on complex mixes
```

---

### Feature Extraction

All audio features are extracted using librosa with consistent parameters:

```python
# Common Parameters
SAMPLE_RATE = 22050
HOP_LENGTH  = 1024  # ~46ms per frame
N_FFT       = 2048
N_MELS      = 128
N_MFCC      = 20

# 43-Dimensional Feature Vector (per frame)
features = [
    mfcc[0:13],            # 13 dims - Timbre
    spectral_centroid,     #  1 dim  - Brightness
    spectral_bandwidth,    #  1 dim  - Spread
    spectral_rolloff,      #  1 dim  - High-freq energy
    spectral_contrast[7],  #  7 dims - Harmonic structure
    spectral_flatness,     #  1 dim  - Noise vs tone
    zero_crossing_rate,    #  1 dim  - Percussive content
    rms_energy,            #  1 dim  - Loudness
    onset_strength,        #  1 dim  - Transient detection
    chroma[0:12],          # 12 dims - Pitch class
    tempo_feature,         #  1 dim  - BPM context
    beat_sync,             #  1 dim  - Beat alignment
    delta_mfcc[0:1],       #  1 dim  - Temporal change
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
│       │   ├── beat_generator.py     # StableBeaT generative pipeline
│       │   └── mixer.py              # DJ mixing engine
│       │
│       ├── model/                    # ML Components
│       │   ├── ann_model.py          # ANN architecture
│       │   ├── train.py              # Training script
│       │   ├── inference.py          # ANN inference
│       │   └── stable_beat/          # StableBeaT LoRA weights + config
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
│           ├── output/               # Generated mixes & beats
│           └── models/               # Trained model weights
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
│               ├── midi_patterns.pkl
│               └── stable_beat/      # StableBeaT LoRA checkpoint
│
├── frontend/                         # React Frontend
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── api.js
│       ├── index.css
│       ├── components/
│       │   ├── WaveformPlayer.jsx
│       │   ├── StatCard.jsx
│       │   └── GenreBar.jsx
│       └── pages/
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
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- FFmpeg (for audio processing)
- CUDA-capable GPU recommended for beat generation inference

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

# Place model weights in backend/app/storage/models/
# ANN models: .h5 and .pkl files
# StableBeaT: LoRA checkpoint in stable_beat/
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Running the Application
```bash
# Terminal 1: Backend
cd backend
uvicorn app.main:app --reload --port 8002

# Terminal 2: Frontend
cd frontend
npm run dev
```

Access the application at **http://localhost:5173**

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload audio file |
| POST | `/analyze` | Analyze uploaded track |
| POST | `/mix` | Mix two tracks |
| POST | `/generate` | Generate beat from prompt |
| GET | `/output/{file_id}` | Download output file |

---

### POST /upload
```bash
curl -X POST "http://localhost:8002/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@track.mp3"
```
```json
{
  "file_id": "abc123-def456-ghi789",
  "filename": "track.mp3"
}
```

---

### POST /analyze
```json
// Request
{ "file_id": "abc123-def456-ghi789" }

// Response
{
  "bpm": 128.5,
  "beat_times": [0.45, 0.92, 1.38, 1.85],
  "duration": 180.5,
  "genre": "hiphop",
  "genre_confidence": 0.87,
  "genre_top3": [
    {"genre": "hiphop", "confidence": 0.87},
    {"genre": "pop",    "confidence": 0.08},
    {"genre": "jazz",   "confidence": 0.03}
  ],
  "dominant_instrument": "keyboard",
  "instrument_confidence": 0.72,
  "instruments_top3": [
    {"instrument": "keyboard",   "confidence": 0.72},
    {"instrument": "bass",       "confidence": 0.15},
    {"instrument": "synth_lead", "confidence": 0.08}
  ],
  "tags": ["piano", "drums", "slow", "ambient"],
  "tag_scores": [
    {"tag": "piano", "score": 0.89},
    {"tag": "drums", "score": 0.76}
  ],
  "mood":       "calm",
  "has_vocals": false,
  "energy":     "medium"
}
```

---

### POST /mix
```json
// Request
{
  "track_a_id": "file-id-1",
  "track_b_id": "file-id-2",
  "crossfade_duration": 8.0
}

// Response
{
  "output_file_id": "mix-xyz789",
  "duration": 240.5,
  "target_bpm": 126.5
}
```

---

### POST /generate

Generate a full beat using the StableBeaT generative pipeline. Prompts are enriched via Llama 3.1 3B before being passed to the fine-tuned model.

```json
// Request
{
  "prompt": "Create a 120 BPM trap beat with heavy 808s, 4 bars, dark mood"
}

// Response
{
  "output_file_id": "beat-abc123",
  "bpm": 120,
  "bars": 4,
  "genre": "trap",
  "key": "C minor",
  "enriched_prompt": "A dark trap beat at 120 BPM in C minor, featuring heavy 808 bass, synth bells, and a melancholic atmosphere.",
  "duration_seconds": 47,
  "generation_steps": 200
}
```

---

## Models

### Model Architectures

#### Beat Detector
```
Input:  43-dim feature vector
Dense(128, relu) → Dropout(0.3)
Dense(64,  relu) → Dropout(0.3)
Dense(1, sigmoid)
Output: P(beat) ∈ [0, 1]
```

#### Genre Classifier
```
Input:  57-dim feature vector
Dense(256, relu) → BatchNorm → Dropout(0.4)
Dense(128, relu) → BatchNorm → Dropout(0.4)
Dense(10, softmax)
Output: 10-class probabilities
```

#### Instrument Classifier
```
Input:  43-dim feature vector
Dense(128, relu) → Dropout(0.3)
Dense(64,  relu) → Dropout(0.3)
Dense(11, softmax)
Output: 11-family probabilities
```

#### Tag Predictor
```
Input:  43-dim feature vector
Dense(256, relu) → Dropout(0.4)
Dense(128, relu) → Dropout(0.4)
Dense(56, sigmoid)
Output: Multi-label scores (threshold: 0.3)
```

#### StableBeaT (Beat Generator)
```
Base:      Stable Audio Open 1.0
Fine-tune: LoRA (via LoRAW pipeline)
Dataset:   40,000 trap/rap/R&B audio segments (~277h)
Epochs:    14   |   Batch size: 16   |   Steps: ~35,000
Hardware:  NVIDIA A100 (Google Colab, ~42h training)
Inference: ~1m 15s on RTX 4050 Laptop GPU
Settings:  200 steps, CFG scale 7, duration up to 47s
```

### StableBeaT Generation Examples

All examples below were generated with 200 steps, CFG scale 7, duration 47s.

| Prompt | BPM | Spectral Centroid | H/P Ratio | CLAP Score |
|---|---|---|---|---|
| Dark melancholic cloud trap, nostalgic piano, plucked bass, synth bells, 110 BPM | 106.13 | 1159.43 | 0.460 | 0.489 |
| Lo-fi jazz rap, 85 BPM, deep sub, plucked bass, vocal chop, chill jazzy mood | 82.72 | 784.82 | 0.457 | 0.429 |
| Melancholic trap, 105 BPM, synth bells, deep sub, minor piano, airy vocal pads | 100.45 | 2540.28 | 1.412 | 0.523 |
| Jazzy chillhop, 101 BPM, synth bells, vocal pad, movie sample, nostalgic mood | 148.02 | 4287.26 | 2.963 | 0.552 |
| Smooth trap, 115 BPM, electric guitar, plucked bass, vocal adlibs, warm pads | 82.72 | 1056.42 | 0.645 | 0.478 |
| Moody cloud trap, 100 BPM, boomy bass, synth bells, melodic piano | 144.2 | 2458.5 | 0.738 | 0.363 |
| Neo-soul R&B, 90 BPM, D major, live bass, soft Rhodes, analog drum grooves | 130.81 | 1000.87 | 0.679 | 0.250 |

### StableBeaT Performance Notes

**Strengths:**
- Excels on melodic, atmospheric beats with smooth harmonic coherence
- Strong instrument-mood-tempo consistency; outputs feel musically balanced
- Captures nuanced subgenre characteristics (cloud trap, chillhop, neo-soul)

**Current Limitations:**
- Underperforms on underrepresented styles (boom bap, high-energy dense percussion)
- CLAP LAION tagging not specialized for trap/hip-hop → imprecise labeling of snares, hi-hats, 808s
- Melodic elements (piano, synths) can sound quieter than drums due to frequency range differences

### Training (ANN Models)
```bash
# Prepare beat detection data
python -m backend.app.data.prepare_data

# Train beat detector
python -m backend.app.model.train

# Other models use Kaggle notebooks:
# notebooks/kaggle/GenreClassifier_Training.ipynb
# notebooks/kaggle/MultiTask_Audio_Training.ipynb
```

---

## Configuration

Environment variables (prefix: `AUTOMIX_`):

| Variable | Default | Description |
|---|---|---|
| `AUTOMIX_SAMPLE_RATE` | 22050 | Audio sample rate |
| `AUTOMIX_HOP_LENGTH` | 512 | STFT hop length (training) |
| `AUTOMIX_MODEL_PATH` | storage/models/beat_detector.h5 | Beat model path |
| `AUTOMIX_GENRE_MODEL_PATH` | storage/models/genre_classifier.h5 | Genre model |
| `AUTOMIX_STABLE_BEAT_PATH` | storage/models/stable_beat/ | StableBeaT LoRA path |
| `AUTOMIX_DEBUG` | false | Enable debug logging |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, WaveSurfer.js, Lucide Icons |
| Backend | FastAPI, Uvicorn, Pydantic |
| ML / ANN | TensorFlow/Keras, scikit-learn, NumPy |
| Beat Generation | Stable Audio Open 1.0, LoRAW, Stable Audio Tools |
| Prompt Enrichment | Llama 3.1 3B (local inference) |
| Audio Tagging | CLAP LAION model |
| Feature Extraction | librosa, Essentia |
| Audio I/O | soundfile, pyrubberband |
| Storage | File-based (uploads/, output/) |

---

## Performance

| Task | Metric | Value |
|---|---|---|
| Beat Detection | F1-score | ~0.85 |
| BPM Estimation | Accuracy (±2 BPM) | ~92% |
| Genre Classification | Accuracy (10-class) | ~78% |
| Instrument Classification | Accuracy (11-class) | ~71% |
| Tag Prediction | mAP@10 | ~0.68 |
| Beat Generation (StableBeaT) | CLAP Prompt Score (avg) | ~0.44 |
| Beat Generation | Inference time (RTX 4050) | ~1m 15s |

---

## Roadmap

- [ ] Real-time streaming analysis
- [ ] Key detection for harmonic mixing
- [ ] Energy-based track matching
- [ ] Playlist-level intelligent mixing
- [ ] Reinforcement learning DJ transitions
- [ ] Trap-specialized CLAP model for improved tagging
- [ ] StableBeaT fine-tuning on underrepresented styles (boom bap, high-energy)
- [ ] SpecGrad-style noise conditioning for beat generation
- [ ] Mobile app (React Native)
- [ ] Cloud deployment (AWS/GCP)

---

## License

MIT License — see `LICENSE` for details.

---

## Acknowledgments

- [librosa](https://librosa.org) — Audio analysis library
- [TensorFlow](https://tensorflow.org) — Machine learning framework
- [FastAPI](https://fastapi.tiangolo.com) — Modern web framework
- [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) — Generative audio foundation model
- [LoRAW](https://github.com/NeuralNotW0rk/LoRAW) — LoRA fine-tuning pipeline for Stable Audio Open
- [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools) — Official Stability AI audio framework
- [Essentia](https://essentia.upf.edu/models.html) — Music feature extraction
- [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) — Genre classification
- [NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth) — Instrument sounds
- [MagnaTagATune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) — Music tagging
