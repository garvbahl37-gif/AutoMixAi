---
title: AutoMixAI Beat Generator
emoji: 🥁
colorFrom: purple
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
suggested_hardware: t4-small
---

# AutoMixAI Beat Generator

AI-powered beat and music generation using **Meta's MusicGen** model.

Generate studio-quality beats, loops, and music from natural language text prompts.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/generate` | Generate beat from text prompt |
| GET | `/output/{id}` | Download generated audio |

## Generate Request

```json
{
  "prompt": "hard-hitting trap beat with 808 bass and rolling hi-hats",
  "duration": 10,
  "temperature": 1.0,
  "guidance_scale": 3.0
}
```

## Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `prompt` | 3-500 chars | Natural language beat description |
| `duration` | 3-30 seconds | Length of generated audio |
| `temperature` | 0.5-1.5 | Creativity (lower=predictable, higher=creative) |
| `guidance_scale` | 1.0-10.0 | Prompt adherence (higher=stricter) |

## Model

Uses `facebook/musicgen-small` by default. Set `MUSICGEN_MODEL` env var to change:
- `facebook/musicgen-small` — Fast, decent quality (300M params)
- `facebook/musicgen-medium` — Better quality (1.5B params)
- `facebook/musicgen-large` — Best quality (3.3B params)
