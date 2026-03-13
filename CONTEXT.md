# AutoMixAI – Context for Continuation

## Project Status (as of 2026-02-11 03:00 IST)

### ✅ COMPLETED

**Backend Architecture (fully built)**
- `backend/app/main.py` – FastAPI entry point with CORS, lifespan
- `backend/app/routes/` – upload.py, analyze.py, mix.py (4 endpoints)
- `backend/app/services/` – audio_loader, feature_extractor, beat_detector, bpm_estimator, mixer
- `backend/app/model/` – ann_model.py (Keras Dense ANN), train.py, inference.py
- `backend/app/schemas/` – Pydantic request/response models
- `backend/app/utils/` – config.py (Pydantic Settings), logger.py, helpers.py
- `backend/app/data/prepare_data.py` – Ballroom + FMA dataset processor

**Datasets (extracted & ready)**
- `Datasets/BallroomData/` – 698 WAV files across 10 genre folders
- `Datasets/BallroomAnnotations/ballroomGroundTruth/` – 698 .bpm files
- `Datasets/fma_small/fma_small/` – 851 MP3 files in 156 subdirs
- `Datasets/fma_metadata/fma_metadata/` – tracks.csv, genres.csv, etc.

**Data Prepared**
- `backend/data/processed/X.npy` – 1,124,394 × 16 feature matrix
- `backend/data/processed/y.npy` – binary beat labels (4.5% beat ratio)

**React Frontend (fully built)**
- `frontend/` – Vite + React project
- `frontend/src/index.css` – dark theme design system
- `frontend/src/api.js` – API client for backend
- `frontend/src/App.jsx` – sidebar navigation shell
- `frontend/src/pages/UploadPage.jsx` – drag & drop upload
- `frontend/src/pages/AnalyzePage.jsx` – BPM, beat timeline, stats
- `frontend/src/pages/MixPage.jsx` – dual deck mixer, crossfade, preview

**Kaggle Notebook**
- `kaggle/AutoMixAI_Training.ipynb` – self-contained training notebook

**Python 3.11 venv (for TensorFlow)**
- `backend/venv311/` – Python 3.11 virtualenv with TF + all deps installed

### ⏳ IN PROGRESS

**Model Training**
- Running locally: epoch 6/50, val_accuracy: 95.68%
- ~60s/epoch → ~45 min remaining
- Model auto-saves to `backend/app/storage/models/beat_detector.h5`
- If training was interrupted, restart with:
  ```bash
  cd backend
  venv311\Scripts\python.exe -m app.model.train
  ```
- For full dataset training: upload datasets to Kaggle and use `kaggle/AutoMixAI_Training.ipynb`

## Key Commands

```bash
# Backend (from backend/ dir)
venv311\Scripts\python.exe -m app.data.prepare_data     # prepare training data
venv311\Scripts\python.exe -m app.model.train            # train ANN
venv311\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000  # start API

# Frontend (from frontend/ dir)
npm run dev     # starts on http://localhost:5173

# API docs: http://localhost:8000/docs
```

## File Tree

```
AutoMixAI/
├── CONTEXT.md                     # This file
├── Datasets/                      # Raw datasets (Ballroom + FMA)
├── kaggle/
│   └── AutoMixAI_Training.ipynb   # Kaggle training notebook
├── backend/
│   ├── venv311/                   # Python 3.11 venv (TensorFlow)
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/     (upload, analyze, mix)
│   │   ├── schemas/    (audio_request, analysis_response)
│   │   ├── services/   (audio_loader, feature_extractor, beat_detector, bpm_estimator, mixer)
│   │   ├── model/      (ann_model, train, inference)
│   │   ├── data/       (prepare_data)
│   │   ├── utils/      (config, logger, helpers)
│   │   └── storage/    (uploads, outputs, models)
│   ├── data/           (raw, processed, labels)
│   ├── requirements.txt
│   └── README.md
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── api.js
    │   ├── index.css
    │   ├── main.jsx
    │   └── pages/  (UploadPage, AnalyzePage, MixPage)
    ├── index.html
    └── package.json
```
