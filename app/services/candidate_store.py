import json
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.schemas.candidate import Candidate


@lru_cache(maxsize=8)
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
    """Create a recruiter-style profile string for hybrid retrieval and re-ranking."""
    history_text = ". ".join(
        (
            f"{entry.title} at {entry.company} from {entry.start_date.isoformat()} "
            f"to {(entry.end_date.isoformat() if entry.end_date else 'present')} "
            f"using {', '.join(entry.skills)}"
        ).strip()
        for entry in candidate.role_history
    )

    parts = [
        f"Candidate {candidate.id}",
        candidate.role_title,
        candidate.seniority,
        f"{candidate.total_experience_years} years of experience",
        f"Current company: {candidate.current_company or 'unknown'}",
        f"Skills: {', '.join(candidate.skills)}",
        f"Preferred roles: {', '.join(candidate.preferred_roles)}",
        f"Industries: {', '.join(candidate.industries)}",
        f"Education: {candidate.education}",
        f"Work preference: {candidate.work_preference}",
        f"Role history: {history_text}",
        f"Profile summary: {candidate.profile_summary}",
    ]
    return ". ".join(part for part in parts if part and not part.endswith(": "))


def build_candidate_skill_text(candidate: Candidate) -> str:
    history_skills = []
    for entry in candidate.role_history:
        history_skills.extend(entry.skills)

    parts = [
        f"Primary skills: {', '.join(candidate.skills)}",
        f"Historical skills: {', '.join(history_skills)}",
        f"Role keywords: {candidate.role_title}, {', '.join(candidate.preferred_roles)}",
    ]
    return ". ".join(part for part in parts if part)
