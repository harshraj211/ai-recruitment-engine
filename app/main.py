import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import get_settings


def ensure_runtime_dirs() -> None:
    for relative_path in (
        "data/candidates",
        "data/conversations",
        "data/faiss",
    ):
        Path(relative_path).mkdir(parents=True, exist_ok=True)


def configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    ensure_runtime_dirs()
    yield


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Hackathon-ready backend for job description parsing, candidate matching, "
        "and AI-assisted engagement."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/api")
async def api_info() -> dict[str, str]:
    return {
        "message": "Talent scouting backend is running.",
        "docs_url": "/docs",
        "health_url": f"{settings.api_v1_prefix}/health",
        "match_url": f"{settings.api_v1_prefix}/match",
        "stream_match_url": f"{settings.api_v1_prefix}/match/stream",
        "stage": "step_9_api_ready",
    }


# Mount frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
