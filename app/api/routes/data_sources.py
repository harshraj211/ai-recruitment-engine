import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import get_settings
from app.schemas.candidate import Candidate
from app.services.candidate_store import (
    get_data_source_status,
    load_candidates,
    use_candidate_dataset,
    use_local_dataset,
)

router = APIRouter()


class CandidateUploadRequest(BaseModel):
    candidates: list[dict[str, Any]]


def _first_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _parse_number(value: Any, field_name: str, row_number: int) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower().replace(",", "").replace("$", "")
        multiplier = 1.0
        if normalized.endswith("k"):
            multiplier = 1_000.0
            normalized = normalized[:-1]
        elif normalized.endswith("m"):
            multiplier = 1_000_000.0
            normalized = normalized[:-1]
        if re.fullmatch(r"\d+(?:\.\d+)?", normalized):
            return float(normalized) * multiplier
    raise ValueError(f"Row {row_number}: {field_name} must be a number.")


def _normalize_skills(value: Any, row_number: int) -> list[str]:
    if isinstance(value, str):
        skills = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        skills = [str(item).strip() for item in value]
    else:
        raise ValueError(f"Row {row_number}: skills must be a list or comma-separated string.")

    skills = [skill for skill in skills if skill]
    if not skills:
        raise ValueError(f"Row {row_number}: skills must include at least one value.")
    return skills


def _infer_seniority(experience_years: float) -> str:
    if experience_years >= 10:
        return "principal"
    if experience_years >= 8:
        return "lead"
    if experience_years >= 5:
        return "senior"
    if experience_years >= 2:
        return "mid-level"
    return "junior"


def _normalize_candidate(record: dict[str, Any], index: int) -> Candidate:
    row_number = index + 1
    name = _first_value(record, ("name", "full_name"))
    skills = _first_value(record, ("skills",))
    experience = _first_value(record, ("experience", "total_experience_years"))
    salary = _first_value(record, ("salary", "expected_salary_usd"))

    missing_fields = [
        field
        for field, value in (
            ("name", name),
            ("skills", skills),
            ("experience", experience),
            ("salary", salary),
        )
        if value in (None, "")
    ]
    if missing_fields:
        raise ValueError(f"Row {row_number}: missing required fields: {', '.join(missing_fields)}.")

    total_experience_years = _parse_number(experience, "experience", row_number)
    expected_salary_usd = int(_parse_number(salary, "salary", row_number))
    role_title = _first_value(record, ("role_title", "role", "title")) or "Candidate"
    current_company = _first_value(record, ("current_company", "company"))
    location = _first_value(record, ("location",)) or "Unknown"
    candidate_id = str(_first_value(record, ("id", "candidate_id")) or f"upload-{row_number:03d}")
    normalized_skills = _normalize_skills(skills, row_number)

    normalized = {
        **record,
        "id": candidate_id,
        "full_name": str(name).strip(),
        "role_title": str(role_title).strip(),
        "seniority": record.get("seniority") or _infer_seniority(total_experience_years),
        "location": str(location).strip(),
        "total_experience_years": total_experience_years,
        "skills": normalized_skills,
        "preferred_roles": record.get("preferred_roles") or [str(role_title).strip()],
        "industries": record.get("industries") or [],
        "education": record.get("education") or "Not specified",
        "work_preference": record.get("work_preference") or "not specified",
        "current_status": record.get("current_status") or "uploaded",
        "expected_salary_usd": expected_salary_usd,
        "availability_days": record.get("availability_days", 30),
        "profile_summary": record.get("profile_summary")
        or f"{name} has {total_experience_years:g} years of experience with {', '.join(normalized_skills[:5])}.",
        "current_company": current_company,
    }
    return Candidate.model_validate(normalized)


def validate_uploaded_candidates(records: list[dict[str, Any]]) -> list[Candidate]:
    if not records:
        raise ValueError("Candidate upload must include at least one record.")
    candidates = [_normalize_candidate(record, index) for index, record in enumerate(records)]
    candidate_ids = [candidate.id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate ids must be unique.")
    return candidates


@router.get("/data-source")
async def data_source_status() -> dict[str, object]:
    return get_data_source_status()


@router.post("/data-source/local")
async def select_local_dataset() -> dict[str, object]:
    status = use_local_dataset()
    return {
        **status,
        "detail": f"Using {status['candidate_count']} candidates from {get_settings().candidate_data_path}.",
    }


@router.post("/data-source/upload")
async def upload_candidate_dataset(payload: CandidateUploadRequest) -> dict[str, object]:
    try:
        candidates = validate_uploaded_candidates(payload.candidates)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    status = use_candidate_dataset(candidates, "upload")
    return {
        **status,
        "detail": f"Uploaded and activated {status['candidate_count']} candidate profiles.",
    }


@router.get("/mock-candidates")
async def mock_candidates() -> list[dict[str, object]]:
    return [candidate.model_dump(mode="json") for candidate in load_candidates(get_settings().candidate_data_path)]


@router.post("/data-source/mock-api")
async def select_mock_api_dataset() -> dict[str, object]:
    candidates = load_candidates(get_settings().candidate_data_path)
    status = use_candidate_dataset(candidates, "mock_api")
    return {
        **status,
        "detail": f"Loaded {status['candidate_count']} candidates from the simulated external API.",
    }
