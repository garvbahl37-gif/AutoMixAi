# AutoMixAI – Backend

AutoMixAI is an AI-powered backend system for automated DJ mixing. The system analyzes music tracks, detects rhythmic structure using neural networks, estimates tempo, and produces beat-synchronized DJ-style mixes.

The backend exposes a REST API that allows clients to upload audio tracks, analyze their rhythmic properties, and generate synchronized mixes.

The core capabilities include:

* Beat detection using neural networks
* BPM estimation and tempo tracking
* Beat-aligned track synchronization
* Automated crossfading and DJ-style transitions
* Audio processing using signal processing and machine learning

---

# System Architecture

The backend follows a modular architecture designed for scalability and separation of concerns.

```text
                Client Application
                        │
                        │ HTTP Requests
                        ▼
                FastAPI Backend Server
                        │
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
    Upload Service   Analysis Service   Mixing Service
        │               │                │
        │               │                │
        ▼               ▼                ▼
  Audio Storage     Beat Detection      Mixing Engine
                        │
                        ▼
                 BPM Estimation
                        │
                        ▼
                 Beat Alignment
                        │
                        ▼
                 Output Generation
```

---

# Backend Directory Structure

```text
backend/
├── app/
│
│   ├── main.py
│   │
│   │   FastAPI application entry point
│   │   Initializes routes, middleware, and configuration
│
│   ├── routes/
│   │
│   │   API endpoint definitions
│   │
│   │   ├── upload.py
│   │   │   Handles audio file uploads
│   │   │
│   │   ├── analyze.py
│   │   │   Performs BPM and beat analysis
│   │   │
│   │   └── mix.py
│   │       Generates automated DJ mixes
│
│   ├── schemas/
│   │
│   │   Pydantic request and response models
│   │
│   │   ├── audio_schema.py
│   │   └── mix_schema.py
│
│   ├── services/
│   │
│   │   Core audio processing modules
│   │
│   │   ├── audio_loader.py
│   │   │   Handles audio decoding and loading
│   │   │
│   │   ├── beat_detector.py
│   │   │   Neural network based beat detection
│   │   │
│   │   ├── bpm_estimator.py
│   │   │   Tempo estimation and rhythm analysis
│   │   │
│   │   ├── track_alignment.py
│   │   │   Synchronizes tracks based on detected beats
│   │   │
│   │   └── auto_mixer.py
│   │       Generates crossfades and DJ transitions
│
│   ├── model/
│   │
│   │   Machine learning components
│   │
│   │   ├── model.py
│   │   │   Neural network architecture
│   │   │
│   │   ├── train.py
│   │   │   Training pipeline
│   │   │
│   │   └── inference.py
│   │       Model inference logic
│
│   ├── utils/
│   │
│   │   Shared utilities
│   │
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── file_utils.py
│
│   └── storage/
│       │
│       ├── uploads/
│       │   Temporary storage for uploaded tracks
│       │
│       ├── output/
│       │   Generated mixes
│       │
│       └── models/
│           Trained neural network models
│
├── data/
│   │
│   ├── raw/
│   │   Training audio files
│   │
│   └── labels/
│       Beat annotation CSV files
│
├── requirements.txt
└── README.md
```

---

# Technology Stack

| Component            | Technology                |
| -------------------- | ------------------------- |
| Programming Language | Python 3.10+              |
| Web Framework        | FastAPI                   |
| Machine Learning     | TensorFlow / Keras        |
| Audio Processing     | librosa, scipy, soundfile |
| Validation           | Pydantic                  |
| ASGI Server          | Uvicorn                   |

---

# Audio Processing Pipeline

The backend processes audio through several stages.

```text
        Audio Upload
              │
              ▼
        Audio Preprocessing
      (resampling, normalization)
              │
              ▼
        Feature Extraction
      (Mel Spectrogram, MFCC)
              │
              ▼
        Beat Detection Model
              │
              ▼
        BPM Estimation
              │
              ▼
        Beat Alignment
              │
              ▼
        Mixing Engine
              │
              ▼
        Generated DJ Mix
```

---

# Neural Network Beat Detection

The beat detection system uses a neural network trained on rhythmic audio features.

Input features include:

* Mel spectrogram
* MFCC coefficients
* spectral flux
* onset strength

Model architecture:

```text
Input: Mel Spectrogram
        │
        ▼
Convolution Layer
        │
        ▼
Batch Normalization
        │
        ▼
ReLU Activation
        │
        ▼
Pooling Layer
        │
        ▼
Dense Layer
        │
        ▼
Sigmoid Output
```

The model predicts beat activation probabilities across time frames.

---

# Mixing Engine

The mixing engine aligns tracks and generates transitions.

```text
Track A               Track B
   │                     │
   ▼                     ▼
Beat Detection      Beat Detection
   │                     │
   └────── BPM Sync ─────┘
            │
            ▼
      Beat Alignment
            │
            ▼
       Crossfade Engine
            │
            ▼
       Mixed Output Track
```

Key components:

* tempo synchronization
* beat alignment
* transition generation
* crossfade mixing

---

# Installation

Clone the repository.

```bash
git clone https://github.com/yourusername/automixai.git
cd automixai/backend
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Running the Backend

Start the API server.

```bash
uvicorn app.main:app --reload
```

Server address:

```
http://localhost:8000
```

Interactive API documentation:

```
http://localhost:8000/docs
```

---

# API Endpoints

| Method | Endpoint            | Description                    |
| ------ | ------------------- | ------------------------------ |
| POST   | `/upload`           | Upload an audio file           |
| POST   | `/analyze`          | Detect BPM and beat timestamps |
| POST   | `/mix`              | Generate beat-aligned DJ mix   |
| GET    | `/output/{file_id}` | Download generated mix         |

---

# Model Training

Place training data in the following directories.

```text
data/
 ├── raw/
 │   audio files
 │
 └── labels/
     beat annotations
```

Run the training pipeline.

```bash
python -m app.model.train
```

Trained models will be saved to:

```text
app/storage/models/
```

---

# Output Files

Generated mixes are stored in:

```text
app/storage/output/
```

Files can be retrieved using the `/output/{file_id}` endpoint.

---

# Future Improvements

Potential extensions include:

* harmonic mixing using key detection
* energy based track matching
* transition point detection
* playlist level mixing
* real-time streaming mixing
* reinforcement learning based DJ transitions
