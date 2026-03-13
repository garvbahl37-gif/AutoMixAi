"""
AutoMixAI – FastAPI Application Entry Point

Creates the FastAPI application, registers route modules, configures
CORS, and ensures required storage directories exist on startup.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import upload, analyze, mix
from app.utils.helpers import ensure_directories
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run setup tasks before the server starts accepting requests."""
    logger.info("AutoMixAI starting up — ensuring directories …")
    ensure_directories()
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ─────────────────────────────────────────────────
app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(mix.router)


@app.get("/", tags=["Health"])
async def root():
    """Health-check endpoint."""
    return {"status": "ok", "service": "AutoMixAI"}
