---
title: AutoMixAI Backend
emoji: 🎧
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# AutoMixAI Backend

All-in-one FastAPI backend for AutoMixAI — AI-powered DJ mixing platform.

## Features

- **Upload** — Upload audio files for processing
- **Analyze** — BPM detection, beat tracking, energy analysis, genre estimation
- **Mix** — Advanced DJ mixing with EQ crossfade, LUFS normalization, time-stretching
- **Generate** — Procedural drum beat generation from text prompts
- **Recognize** — Song recognition via Shazam API (RapidAPI)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/upload` | Upload audio file |
| POST | `/analyze` | Analyze audio (BPM, beats, genre) |
| POST | `/mix` | DJ mix two tracks |
| POST | `/generate` | Generate drum beats |
| GET | `/output/{id}` | Download generated audio |
| POST | `/recognize` | Shazam song recognition |

## DJ Mixing Engine

The mixer uses professional techniques:
- LUFS loudness normalization (EBU R128)
- High-pass filtering (40Hz rumble removal)
- Beat-aligned time-stretching
- EQ-based crossfade (bass swap transition)
- Equal-power S-curve crossfade
- Final mastering to -14 LUFS (streaming standard)
