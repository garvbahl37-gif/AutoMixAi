# AutoMixAI – Backend

AI-powered automated DJ mixing system with ANN-based beat detection, BPM estimation, and beat-synchronized audio mixing.

## Architecture

```
backend/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── routes/               # API endpoints
│   ├── schemas/              # Pydantic request/response models
│   ├── services/             # Audio processing & mixing logic
│   ├── model/                # ANN model (build, train, inference)
│   ├── utils/                # Configuration, logging, helpers
│   └── storage/              # Runtime file storage
├── data/                     # Training data
├── requirements.txt
└── README.md
```

## Tech Stack

| Component          | Technology              |
|--------------------|-------------------------|
| Language           | Python 3.10+            |
| Web Framework      | FastAPI                 |
| ML Framework       | TensorFlow / Keras      |
| Audio Processing   | librosa, scipy, soundfile |
| Validation         | Pydantic                |

## Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API server starts at `http://localhost:8000`. Interactive docs at `/docs`.

## API Endpoints

| Method | Endpoint             | Description                          |
|--------|----------------------|--------------------------------------|
| POST   | `/upload`            | Upload an audio file                 |
| POST   | `/analyze`           | Analyze BPM & beat timestamps        |
| POST   | `/mix`               | Generate a beat-synchronized DJ mix  |
| GET    | `/output/{file_id}`  | Download the mixed output file       |

## Training

Place audio files in `data/raw/` and beat annotation CSVs in `data/labels/`. Then:

```bash
python -m app.model.train
```

Trained models are saved to `app/storage/models/`.
