---
title: AutoMixAI Beat Generator
emoji: 🥁
colorFrom: purple
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# AutoMixAI Beat Generator

AI-powered beat and music generation using **Meta's MusicGen** model.

Generate studio-quality beats, loops, and music from natural language text prompts.

> Runs on free CPU tier. Generation takes ~60-90 seconds for 10s of audio.

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
