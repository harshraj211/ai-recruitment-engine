from pathlib import Path

from fastapi import APIRouter

from app.core.config import get_settings
from app.services.candidate_store import load_candidates

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, object]:
    settings = get_settings()
    checks: dict[str, dict[str, str]] = {}

    candidate_file = Path(settings.candidate_data_path)
    try:
        candidate_count = len(load_candidates())
        checks["candidate_data"] = {
            "status": "ok",
            "detail": f"Loaded {candidate_count} candidate profiles from {candidate_file}.",
        }
    except Exception as exc:
        checks["candidate_data"] = {
            "status": "failed",
            "detail": f"Candidate data could not be loaded: {exc}",
        }

    faiss_index_path = Path(settings.faiss_index_path)
    profile_index_path = faiss_index_path.with_name(f"{faiss_index_path.stem}.profile.index")
    skill_index_path = faiss_index_path.with_name(f"{faiss_index_path.stem}.skills.index")
    metadata_path = faiss_index_path.with_suffix(".meta.json")
    has_hybrid_index = all(path.exists() for path in (profile_index_path, skill_index_path, metadata_path))
    checks["vector_index"] = {
        "status": "ok" if has_hybrid_index else "warning",
        "detail": (
            "Hybrid retrieval index files are present."
            if has_hybrid_index
            else "Hybrid retrieval index files are missing and will be rebuilt on demand."
        ),
    }

    checks["llm"] = {
        "status": "ok" if settings.groq_api_key else "fallback",
        "detail": (
            f"Groq model {settings.groq_model} configured."
            if settings.groq_api_key
            else "Groq API key not configured; deterministic LLM fallback is active."
        ),
    }

    overall_status = (
        "ok"
        if checks["candidate_data"]["status"] == "ok"
        else "degraded"
    )
    return {
        "status": overall_status,
        "app_name": settings.app_name,
        "version": settings.app_version,
        "checks": checks,
    }
