import re

from app.schemas.candidate import Candidate


def mask_name(full_name: str) -> str:
    initials = [part[0].upper() for part in full_name.split() if part]
    return ".".join(initials) if initials else "Candidate"


def mask_candidate_payload(candidate: Candidate) -> dict:
    return {
        "candidate_id": candidate.id,
        "candidate_label": mask_name(candidate.full_name),
        "role_title": candidate.role_title,
        "seniority": candidate.seniority,
        "skills": candidate.skills,
        "preferred_roles": candidate.preferred_roles,
        "industries": candidate.industries,
        "work_preference": candidate.work_preference,
        "current_status": candidate.current_status,
        "expected_salary_usd": candidate.expected_salary_usd,
        "availability_days": candidate.availability_days,
        "profile_summary": re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "[NAME]", candidate.profile_summary),
    }
