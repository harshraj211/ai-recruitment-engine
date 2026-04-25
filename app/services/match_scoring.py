import re

from app.schemas.candidate import Candidate
from app.schemas.job_description import ParsedJobDescription
from app.schemas.match_scoring import CandidateMatchResult

DEFAULT_SKILLS_WEIGHT = 0.7
DEFAULT_EXPERIENCE_WEIGHT = 0.3


def normalize_value(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+]+", " ", value.lower())).strip()


def unique_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_items: list[str] = []

    for item in items:
        key = normalize_value(item)
        if key and key not in seen:
            seen.add(key)
            unique_items.append(item)

    return unique_items


def calculate_skill_match(
    parsed_jd: ParsedJobDescription,
    candidate: Candidate,
) -> tuple[float, list[str], list[str]]:
    jd_skills = unique_preserving_order(parsed_jd.skills)

    if not jd_skills:
        return 1.0, [], []

    candidate_skill_keys = {normalize_value(skill) for skill in candidate.skills}
    matched_skills: list[str] = []
    missing_skills: list[str] = []

    for skill in jd_skills:
        if normalize_value(skill) in candidate_skill_keys:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    score = len(matched_skills) / len(jd_skills)
    return score, matched_skills, missing_skills


def calculate_experience_match(
    parsed_jd: ParsedJobDescription,
    candidate: Candidate,
) -> float:
    required_experience = parsed_jd.min_experience_years

    if required_experience is None or required_experience <= 0:
        return 1.0

    return min(candidate.total_experience_years / required_experience, 1.0)


def build_match_explanation(
    candidate: Candidate,
    parsed_jd: ParsedJobDescription,
    match_score: float,
    skill_match_score: float,
    experience_match_score: float,
    matched_skills: list[str],
    missing_skills: list[str],
    skills_weight: float,
    experience_weight: float,
) -> str:
    jd_skill_count = len(unique_preserving_order(parsed_jd.skills))
    skill_summary = (
        f"Skill match {skill_match_score * 100:.1f}% "
        f"({len(matched_skills)}/{jd_skill_count} skills matched)"
        if jd_skill_count
        else "Skill match 100.0% (no explicit JD skills were extracted)"
    )

    if matched_skills:
        skill_summary += f": {', '.join(matched_skills)}"

    if missing_skills:
        skill_summary += f". Missing: {', '.join(missing_skills)}"

    if parsed_jd.min_experience_years is None:
        experience_summary = (
            f"Experience match 100.0% "
            f"(no minimum experience requirement was extracted)"
        )
    else:
        experience_summary = (
            f"Experience match {experience_match_score * 100:.1f}% "
            f"({candidate.total_experience_years:.1f} yrs vs "
            f"{parsed_jd.min_experience_years:.1f} yrs required)"
        )

    final_summary = (
        f"Final Match Score {match_score:.1f}% "
        f"using skills weight {skills_weight:.1f} and experience weight {experience_weight:.1f}"
    )

    return f"{skill_summary}. {experience_summary}. {final_summary}."


def score_candidate_match(
    parsed_jd: ParsedJobDescription,
    candidate: Candidate,
    *,
    skills_weight: float = DEFAULT_SKILLS_WEIGHT,
    experience_weight: float = DEFAULT_EXPERIENCE_WEIGHT,
    semantic_similarity_score: float | None = None,
) -> CandidateMatchResult:
    skill_match_score, matched_skills, missing_skills = calculate_skill_match(parsed_jd, candidate)
    experience_match_score = calculate_experience_match(parsed_jd, candidate)

    total_weight = skills_weight + experience_weight
    match_score = (
        ((skills_weight * skill_match_score) + (experience_weight * experience_match_score))
        / total_weight
    ) * 100

    explanation = build_match_explanation(
        candidate=candidate,
        parsed_jd=parsed_jd,
        match_score=match_score,
        skill_match_score=skill_match_score,
        experience_match_score=experience_match_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        skills_weight=skills_weight,
        experience_weight=experience_weight,
    )

    return CandidateMatchResult(
        candidate_id=candidate.id,
        full_name=candidate.full_name,
        role_title=candidate.role_title,
        total_experience_years=candidate.total_experience_years,
        match_score=round(match_score, 2),
        skill_match_score=skill_match_score,
        experience_match_score=experience_match_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        explanation=explanation,
        semantic_similarity_score=semantic_similarity_score,
    )


def rank_candidates_by_match(
    parsed_jd: ParsedJobDescription,
    candidates: list[Candidate],
    *,
    skills_weight: float = DEFAULT_SKILLS_WEIGHT,
    experience_weight: float = DEFAULT_EXPERIENCE_WEIGHT,
    similarity_lookup: dict[str, float] | None = None,
) -> list[CandidateMatchResult]:
    similarity_lookup = similarity_lookup or {}
    results = [
        score_candidate_match(
            parsed_jd,
            candidate,
            skills_weight=skills_weight,
            experience_weight=experience_weight,
            semantic_similarity_score=similarity_lookup.get(candidate.id),
        )
        for candidate in candidates
    ]

    return sorted(
        results,
        key=lambda item: (
            item.match_score,
            item.semantic_similarity_score if item.semantic_similarity_score is not None else -1.0,
            item.total_experience_years,
        ),
        reverse=True,
    )
