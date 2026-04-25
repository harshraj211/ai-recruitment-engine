import json
from pathlib import Path

from app.core.config import get_settings
from app.schemas.candidate import Candidate


def load_candidates(data_path: str | None = None) -> list[Candidate]:
    """Load and validate candidate records from the local JSON dataset."""
    settings = get_settings()
    candidate_file = Path(data_path or settings.candidate_data_path)
    payload = json.loads(candidate_file.read_text(encoding="utf-8"))
    return [Candidate.model_validate(item) for item in payload]


def load_candidate_lookup(data_path: str | None = None) -> dict[str, Candidate]:
    """Load candidates as an id -> candidate lookup table."""
    return {candidate.id: candidate for candidate in load_candidates(data_path=data_path)}


def build_candidate_search_text(candidate: Candidate) -> str:
    """Create a plain text profile string for later embedding and FAISS search."""
    parts = [
        candidate.role_title,
        candidate.seniority,
        f"{candidate.total_experience_years} years of experience",
        f"Skills: {', '.join(candidate.skills)}",
        f"Preferred roles: {', '.join(candidate.preferred_roles)}",
        f"Industries: {', '.join(candidate.industries)}",
        f"Education: {candidate.education}",
        f"Work preference: {candidate.work_preference}",
        f"Profile summary: {candidate.profile_summary}",
    ]
    return ". ".join(part for part in parts if part)
