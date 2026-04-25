import re
from dataclasses import dataclass

from app.schemas.candidate import Candidate
from app.schemas.job_description import ParsedJobDescription
from app.schemas.match_scoring import CandidateMatchResult

DEFAULT_SKILLS_WEIGHT = 0.70
DEFAULT_EXPERIENCE_WEIGHT = 0.25
DEFAULT_ROLE_WEIGHT = 0.05
CORE_SKILL_WEIGHT = 2.0
SECONDARY_SKILL_WEIGHT = 1.0

ROLE_FAMILY_KEYWORDS = {
    "machine learning": {"machine", "learning", "ml", "ai", "mlops", "scientist"},
    "ml": {"machine", "learning", "ml", "ai", "mlops", "scientist"},
    "ai": {"ai", "ml", "machine", "learning", "generative", "applied"},
    "backend": {"backend", "api", "platform", "python", "java", "microservices"},
    "frontend": {"frontend", "front", "react", "typescript", "ui"},
    "data": {"data", "analytics", "analyst", "scientist", "engineer"},
    "devops": {"devops", "sre", "cloud", "platform", "infrastructure"},
}


@dataclass(frozen=True)
class SkillMatchBreakdown:
    score: float
    core_skill_score: float
    secondary_skill_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    matched_core_skills: list[str]
    missing_core_skills: list[str]
    matched_secondary_skills: list[str]
    missing_secondary_skills: list[str]


def normalize_value(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+]+", " ", value.lower())).strip()


def tokenize_value(value: str) -> set[str]:
    return {token for token in normalize_value(value).split() if len(token) > 1}


def unique_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_items: list[str] = []

    for item in items:
        key = normalize_value(item)
        if key and key not in seen:
            seen.add(key)
            unique_items.append(item)

    return unique_items


def get_jd_skill_groups(parsed_jd: ParsedJobDescription) -> tuple[list[str], list[str]]:
    jd_skills = unique_preserving_order(parsed_jd.skills)
    if not jd_skills:
        return [], []

    core_skills = unique_preserving_order(parsed_jd.core_skills)
    core_keys = {normalize_value(skill) for skill in core_skills}

    if parsed_jd.secondary_skills:
        secondary_skills = unique_preserving_order(parsed_jd.secondary_skills)
    else:
        secondary_skills = [skill for skill in jd_skills if normalize_value(skill) not in core_keys]

    known_keys = {normalize_value(skill) for skill in core_skills + secondary_skills}
    secondary_skills.extend(skill for skill in jd_skills if normalize_value(skill) not in known_keys)
    return core_skills, unique_preserving_order(secondary_skills)


def _score_skill_group(
    required_skills: list[str],
    candidate_skill_keys: set[str],
) -> tuple[float, list[str], list[str]]:
    if not required_skills:
        return 1.0, [], []

    matched = []
    missing = []
    for skill in required_skills:
        if normalize_value(skill) in candidate_skill_keys:
            matched.append(skill)
        else:
            missing.append(skill)

    return len(matched) / len(required_skills), matched, missing


def calculate_weighted_skill_match(
    parsed_jd: ParsedJobDescription,
    candidate: Candidate,
) -> SkillMatchBreakdown:
    core_skills, secondary_skills = get_jd_skill_groups(parsed_jd)
    jd_skills = core_skills + secondary_skills

    if not jd_skills:
        return SkillMatchBreakdown(
            score=1.0,
            core_skill_score=1.0,
            secondary_skill_score=1.0,
            matched_skills=[],
            missing_skills=[],
            matched_core_skills=[],
            missing_core_skills=[],
            matched_secondary_skills=[],
            missing_secondary_skills=[],
        )

    candidate_skill_keys = {normalize_value(skill) for skill in candidate.skills}
    core_score, matched_core, missing_core = _score_skill_group(core_skills, candidate_skill_keys)
    secondary_score, matched_secondary, missing_secondary = _score_skill_group(
        secondary_skills,
        candidate_skill_keys,
    )

    weighted_possible = (len(core_skills) * CORE_SKILL_WEIGHT) + (
        len(secondary_skills) * SECONDARY_SKILL_WEIGHT
    )
    weighted_matched = (len(matched_core) * CORE_SKILL_WEIGHT) + (
        len(matched_secondary) * SECONDARY_SKILL_WEIGHT
    )
    weighted_score = weighted_matched / weighted_possible if weighted_possible else 1.0

    return SkillMatchBreakdown(
        score=weighted_score,
        core_skill_score=core_score,
        secondary_skill_score=secondary_score,
        matched_skills=matched_core + matched_secondary,
        missing_skills=missing_core + missing_secondary,
        matched_core_skills=matched_core,
        missing_core_skills=missing_core,
        matched_secondary_skills=matched_secondary,
        missing_secondary_skills=missing_secondary,
    )


def calculate_skill_match(
    parsed_jd: ParsedJobDescription,
    candidate: Candidate,
) -> tuple[float, list[str], list[str]]:
    breakdown = calculate_weighted_skill_match(parsed_jd, candidate)
    return breakdown.score, breakdown.matched_skills, breakdown.missing_skills


def calculate_experience_match(
    parsed_jd: ParsedJobDescription,
    candidate: Candidate,
) -> float:
    required_experience = parsed_jd.min_experience_years

    if required_experience is None or required_experience <= 0:
        return 1.0

    if candidate.total_experience_years >= required_experience:
        return 1.0

    # Penalize gaps gradually instead of cliffing strong near-fit candidates.
    return max(candidate.total_experience_years / required_experience, 0.0)


def calculate_role_alignment(
    parsed_jd: ParsedJobDescription,
    candidate: Candidate,
) -> float:
    if not parsed_jd.role_title:
        return 1.0

    target = normalize_value(parsed_jd.role_title)
    candidate_roles = [candidate.role_title, *candidate.preferred_roles]
    normalized_roles = [normalize_value(role) for role in candidate_roles]

    if target in normalized_roles:
        return 1.0

    if any(target in role or role in target for role in normalized_roles):
        return 0.85

    target_tokens = tokenize_value(parsed_jd.role_title)
    candidate_tokens = set().union(*(tokenize_value(role) for role in candidate_roles))
    if target_tokens and candidate_tokens:
        token_overlap = len(target_tokens & candidate_tokens) / len(target_tokens)
        if token_overlap >= 0.5:
            return 0.70

    target_family_tokens = set()
    for marker, family_tokens in ROLE_FAMILY_KEYWORDS.items():
        if marker in target:
            target_family_tokens |= family_tokens

    if target_family_tokens and candidate_tokens & target_family_tokens:
        return 0.55

    return 0.25


def build_match_explanation(
    candidate: Candidate,
    parsed_jd: ParsedJobDescription,
    match_score: float,
    skill_breakdown: SkillMatchBreakdown,
    experience_match_score: float,
    role_alignment_score: float,
    skills_weight: float,
    experience_weight: float,
    role_weight: float,
) -> str:
    core_count = len(skill_breakdown.matched_core_skills) + len(skill_breakdown.missing_core_skills)
    secondary_count = len(skill_breakdown.matched_secondary_skills) + len(
        skill_breakdown.missing_secondary_skills
    )

    if core_count or secondary_count:
        skill_summary = (
            f"Weighted skill match {skill_breakdown.score * 100:.1f}% "
            f"(core {len(skill_breakdown.matched_core_skills)}/{core_count}, "
            f"secondary {len(skill_breakdown.matched_secondary_skills)}/{secondary_count})"
        )
    else:
        skill_summary = "Weighted skill match 100.0% (no explicit JD skills were extracted)"

    if skill_breakdown.matched_skills:
        skill_summary += f". Strong matches: {', '.join(skill_breakdown.matched_skills)}"

    if skill_breakdown.missing_core_skills:
        skill_summary += (
            f". Missing critical skills: {', '.join(skill_breakdown.missing_core_skills)}"
        )
    if skill_breakdown.missing_secondary_skills:
        skill_summary += (
            f". Missing secondary skills: {', '.join(skill_breakdown.missing_secondary_skills)}"
        )

    if parsed_jd.min_experience_years is None:
        experience_summary = "Experience fit 100.0% (no minimum experience requirement extracted)"
    else:
        experience_summary = (
            f"Experience fit {experience_match_score * 100:.1f}% "
            f"({candidate.total_experience_years:.1f} yrs vs "
            f"{parsed_jd.min_experience_years:.1f} yrs required)"
        )

    role_summary = (
        f"Role alignment {role_alignment_score * 100:.1f}% "
        f"({candidate.role_title} vs {parsed_jd.role_title or 'unspecified target role'})"
    )

    final_summary = (
        f"Final Match Score {match_score:.1f}% using weights "
        f"skills {skills_weight:.2f}, experience {experience_weight:.2f}, "
        f"role {role_weight:.2f}"
    )

    return f"{skill_summary}. {experience_summary}. {role_summary}. {final_summary}."


def score_candidate_match(
    parsed_jd: ParsedJobDescription,
    candidate: Candidate,
    *,
    skills_weight: float = DEFAULT_SKILLS_WEIGHT,
    experience_weight: float = DEFAULT_EXPERIENCE_WEIGHT,
    role_weight: float = DEFAULT_ROLE_WEIGHT,
    semantic_similarity_score: float | None = None,
) -> CandidateMatchResult:
    skill_breakdown = calculate_weighted_skill_match(parsed_jd, candidate)
    experience_match_score = calculate_experience_match(parsed_jd, candidate)
    role_alignment_score = calculate_role_alignment(parsed_jd, candidate)

    total_weight = skills_weight + experience_weight + role_weight
    match_score = (
        (
            (skills_weight * skill_breakdown.score)
            + (experience_weight * experience_match_score)
            + (role_weight * role_alignment_score)
        )
        / total_weight
    ) * 100

    explanation = build_match_explanation(
        candidate=candidate,
        parsed_jd=parsed_jd,
        match_score=match_score,
        skill_breakdown=skill_breakdown,
        experience_match_score=experience_match_score,
        role_alignment_score=role_alignment_score,
        skills_weight=skills_weight,
        experience_weight=experience_weight,
        role_weight=role_weight,
    )

    return CandidateMatchResult(
        candidate_id=candidate.id,
        full_name=candidate.full_name,
        role_title=candidate.role_title,
        total_experience_years=candidate.total_experience_years,
        match_score=round(match_score, 2),
        skill_match_score=skill_breakdown.score,
        core_skill_score=skill_breakdown.core_skill_score,
        secondary_skill_score=skill_breakdown.secondary_skill_score,
        experience_match_score=experience_match_score,
        role_alignment_score=role_alignment_score,
        matched_skills=skill_breakdown.matched_skills,
        missing_skills=skill_breakdown.missing_skills,
        matched_core_skills=skill_breakdown.matched_core_skills,
        missing_core_skills=skill_breakdown.missing_core_skills,
        matched_secondary_skills=skill_breakdown.matched_secondary_skills,
        missing_secondary_skills=skill_breakdown.missing_secondary_skills,
        explanation=explanation,
        semantic_similarity_score=semantic_similarity_score,
    )


def rank_candidates_by_match(
    parsed_jd: ParsedJobDescription,
    candidates: list[Candidate],
    *,
    skills_weight: float = DEFAULT_SKILLS_WEIGHT,
    experience_weight: float = DEFAULT_EXPERIENCE_WEIGHT,
    role_weight: float = DEFAULT_ROLE_WEIGHT,
    similarity_lookup: dict[str, float] | None = None,
) -> list[CandidateMatchResult]:
    similarity_lookup = similarity_lookup or {}
    results = [
        score_candidate_match(
            parsed_jd,
            candidate,
            skills_weight=skills_weight,
            experience_weight=experience_weight,
            role_weight=role_weight,
            semantic_similarity_score=similarity_lookup.get(candidate.id),
        )
        for candidate in candidates
    ]

    return sorted(
        results,
        key=lambda item: (
            item.match_score,
            item.skill_match_score,
            item.role_alignment_score,
            item.semantic_similarity_score if item.semantic_similarity_score is not None else -1.0,
            item.total_experience_years,
        ),
        reverse=True,
    )
