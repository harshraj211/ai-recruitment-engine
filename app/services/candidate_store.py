import json
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Literal

from app.core.config import get_settings
from app.schemas.candidate import Candidate

DataSourceMode = Literal["local", "upload", "mock_api"]

_DATA_SOURCE_LABELS: dict[DataSourceMode, str] = {
    "local": "Local Dataset",
    "upload": "Upload JSON",
    "mock_api": "Simulated External API",
}
_active_mode: DataSourceMode = "local"
_active_candidates: list[Candidate] | None = None
_source_lock = RLock()


def _load_candidates_from_file(data_path: str | Path) -> list[Candidate]:
    candidate_file = Path(data_path)
    payload = json.loads(candidate_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Candidate dataset must be a JSON array.")
    return [Candidate.model_validate(item) for item in payload]


@lru_cache(maxsize=8)
def load_candidates(data_path: str | None = None) -> list[Candidate]:
    """Load and validate candidate records from the active dataset."""
    if data_path:
        return _load_candidates_from_file(data_path)

    with _source_lock:
        if _active_candidates is not None:
            return [candidate.model_copy(deep=True) for candidate in _active_candidates]

    settings = get_settings()
    return _load_candidates_from_file(settings.candidate_data_path)


def load_candidate_lookup(data_path: str | None = None) -> dict[str, Candidate]:
    """Load candidates as an id -> candidate lookup table."""
    return {candidate.id: candidate for candidate in load_candidates(data_path=data_path)}


def get_data_source_status() -> dict[str, object]:
    """Return the currently selected candidate source for UI display."""
    with _source_lock:
        mode = _active_mode
    return {
        "mode": mode,
        "label": _DATA_SOURCE_LABELS[mode],
        "candidate_count": len(load_candidates()),
    }


def _clear_candidate_dependent_caches() -> None:
    load_candidates.cache_clear()
    try:
        from app.services import jd_parser

        jd_parser.get_skill_aliases.cache_clear()
        jd_parser.get_role_aliases.cache_clear()
        jd_parser.get_domain_aliases.cache_clear()
        jd_parser.get_spacy_components.cache_clear()
    except Exception:
        pass


def _remove_vector_index_files() -> None:
    settings = get_settings()
    index_path = Path(settings.faiss_index_path)
    for path in (
        index_path,
        index_path.with_name(f"{index_path.stem}.profile.index"),
        index_path.with_name(f"{index_path.stem}.skills.index"),
        index_path.with_suffix(".meta.json"),
    ):
        if path.exists():
            path.unlink()


def _activate_candidates(candidates: list[Candidate] | None, mode: DataSourceMode) -> dict[str, object]:
    global _active_candidates, _active_mode
    with _source_lock:
        _active_candidates = [candidate.model_copy(deep=True) for candidate in candidates] if candidates else None
        _active_mode = mode
    _clear_candidate_dependent_caches()
    _remove_vector_index_files()
    return get_data_source_status()


def use_local_dataset() -> dict[str, object]:
    """Switch matching back to the checked-in candidate dataset."""
    return _activate_candidates(None, "local")


def use_candidate_dataset(candidates: list[Candidate], mode: DataSourceMode) -> dict[str, object]:
    """Switch matching to a validated runtime candidate dataset."""
    if mode == "local":
        raise ValueError("Use use_local_dataset() for the local candidate file.")
    return _activate_candidates(candidates, mode)


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
