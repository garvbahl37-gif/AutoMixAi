"""
AutoMixAI – FastAPI Application Entry Point

Creates the FastAPI application, registers route modules, configures
CORS, and ensures required storage directories exist on startup.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import upload, analyze, mix, generate
from app.utils.helpers import ensure_directories
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run setup tasks before the server starts accepting requests."""
    logger.info("AutoMixAI starting up - ensuring directories ...")
    ensure_directories()
    # Pre-warm TensorFlow so it initialises in the worker process before
    # the first request, avoiding a crash when it loads inside a thread pool.
    try:
        from app.model.inference import load_model
        from app.utils.config import settings
        load_model(str(settings.model_path))
        logger.info("TensorFlow model pre-loaded successfully.")
    except FileNotFoundError:
        logger.warning("No trained model found — ANN beat detection unavailable, librosa fallback active.")
    except Exception as exc:
        logger.warning("Could not pre-load model (%s) — librosa fallback active.", exc)
    yield
    logger.info("AutoMixAI shutting down.")


# ── App factory ───────────────────────────────────────────────────────

app = FastAPI(
    title="AutoMixAI",
    description=(
        "AI-powered automated DJ mixing system with ANN-based beat "
        "detection, BPM estimation, and beat-synchronised audio mixing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow any origin for dev; tighten for production) ───────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ─────────────────────────────────────────────────
app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(mix.router)
app.include_router(generate.router)


@app.get("/", tags=["Health"])
async def root():
    """Health-check endpoint."""
    return {"status": "ok", "service": "AutoMixAI"}
