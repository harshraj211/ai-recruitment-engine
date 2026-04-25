import re
from dataclasses import dataclass

from app.schemas.candidate import Candidate
from app.schemas.job_description import ParsedJobDescription
from app.schemas.match_scoring import CandidateMatchResult
from app.services.experience_intelligence import build_skill_recency_weights, career_trajectory_boost
from app.services.skill_graph import SkillGraphService, normalize_skill

DEFAULT_SKILLS_WEIGHT = 0.65
DEFAULT_EXPERIENCE_WEIGHT = 0.20
DEFAULT_ROLE_WEIGHT = 0.15
CORE_SKILL_WEIGHT = 2.0
SECONDARY_SKILL_WEIGHT = 1.0
MIN_SKILL_MATCH_THRESHOLD = 0.55

ROLE_FAMILY_KEYWORDS = {
    "machine learning": {"machine", "learning", "ml", "ai", "mlops", "scientist"},
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
    skill_alignment_details: list[str]


def normalize_value(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+]+", " ", value.lower())).strip()


def tokenize_value(value: str) -> set[str]:
    return {token for token in normalize_value(value).split() if len(token) > 1}


def unique_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = normalize_value(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def get_jd_skill_groups(parsed_jd: ParsedJobDescription) -> tuple[list[str], list[str]]:
    mandatory_skills = unique_preserving_order(parsed_jd.mandatory_skills or parsed_jd.core_skills)
    nice_to_have = unique_preserving_order(parsed_jd.nice_to_have_skills or parsed_jd.secondary_skills)
    if not mandatory_skills and parsed_jd.skills:
        mandatory_skills = unique_preserving_order(parsed_jd.skills[: min(5, len(parsed_jd.skills))])
    known = {normalize_value(skill) for skill in mandatory_skills}
    nice_to_have = [skill for skill in nice_to_have if normalize_value(skill) not in known]
    return mandatory_skills, nice_to_have


def calculate_experience_match(
    parsed_jd: ParsedJobDescription,
    candidate: Candidate,
) -> float:
    required_experience = parsed_jd.min_experience_years
    if required_experience is None or required_experience <= 0:
        return 1.0

    candidate_years = candidate.total_experience_years
    if candidate_years >= required_experience:
        return 1.0

    gap = required_experience - candidate_years
    gap_ratio = gap / required_experience
    if gap_ratio <= 0.10:
        return 0.85
    if gap_ratio <= 0.25:
        return 0.80
    if gap_ratio <= 0.40:
        return 0.60
    return max(0.20, candidate_years / required_experience * 0.5)


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
        return 0.88

    target_tokens = tokenize_value(parsed_jd.role_title)
    candidate_tokens = set().union(*(tokenize_value(role) for role in candidate_roles))
    if target_tokens:
        overlap = len(target_tokens & candidate_tokens) / len(target_tokens)
        if overlap >= 0.5:
            return 0.72

    role_family_tokens = set()
    for marker, family_tokens in ROLE_FAMILY_KEYWORDS.items():
        if marker in target:
            role_family_tokens |= family_tokens
    if role_family_tokens and candidate_tokens & role_family_tokens:
        return 0.55
    return 0.25


def _score_skill_group(
    required_skills: list[str],
    candidate_skills: list[str],
    recency_weights: dict[str, float],
    skill_graph_service: SkillGraphService,
) -> tuple[float, list[str], list[str], list[str]]:
    if not required_skills:
        return 1.0, [], [], []

    matched = []
    missing = []
    details = []
    weighted_scores = []
    for required_skill in required_skills:
        best_skill, similarity = skill_graph_service.best_match(required_skill, candidate_skills)
        if best_skill is None or similarity < MIN_SKILL_MATCH_THRESHOLD:
            missing.append(required_skill)
            details.append(f"{required_skill}: missing")
            weighted_scores.append(0.0)
            continue

        recency = recency_weights.get(best_skill, 0.7)
        evidence_score = similarity * recency
        matched.append(required_skill)
        if normalize_skill(required_skill) == normalize_skill(best_skill):
            details.append(f"{required_skill}: exact ({evidence_score:.2f})")
        else:
            details.append(f"{required_skill}: matched via {best_skill} ({evidence_score:.2f})")
        weighted_scores.append(evidence_score)

    return sum(weighted_scores) / len(required_skills), matched, missing, details


def calculate_weighted_skill_match(
    parsed_jd: ParsedJobDescription,
    candidate: Candidate,
    *,
    skill_graph_service: SkillGraphService | None = None,
) -> SkillMatchBreakdown:
    skill_graph_service = skill_graph_service or SkillGraphService()
    mandatory_skills, nice_to_have_skills = get_jd_skill_groups(parsed_jd)
    if not mandatory_skills and not nice_to_have_skills:
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
            skill_alignment_details=[],
        )

    recency_weights = build_skill_recency_weights(candidate)
    candidate_skills = unique_preserving_order(
        candidate.skills + [skill for entry in candidate.role_history for skill in entry.skills]
    )
    core_score, matched_core, missing_core, core_details = _score_skill_group(
        mandatory_skills,
        candidate_skills,
        recency_weights,
        skill_graph_service,
    )
    secondary_score, matched_secondary, missing_secondary, secondary_details = _score_skill_group(
        nice_to_have_skills,
        candidate_skills,
        recency_weights,
        skill_graph_service,
    )

    weighted_possible = (len(mandatory_skills) * CORE_SKILL_WEIGHT) + (
        len(nice_to_have_skills) * SECONDARY_SKILL_WEIGHT
    )
    weighted_actual = (core_score * len(mandatory_skills) * CORE_SKILL_WEIGHT) + (
        secondary_score * len(nice_to_have_skills) * SECONDARY_SKILL_WEIGHT
    )
    weighted_score = weighted_actual / weighted_possible if weighted_possible else 1.0

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
        skill_alignment_details=core_details + secondary_details,
    )


def calculate_skill_match(
    parsed_jd: ParsedJobDescription,
    candidate: Candidate,
) -> tuple[float, list[str], list[str]]:
    breakdown = calculate_weighted_skill_match(parsed_jd, candidate)
    return breakdown.score, breakdown.matched_skills, breakdown.missing_skills


def build_match_explanation(
    candidate: Candidate,
    parsed_jd: ParsedJobDescription,
    match_score: float,
    skill_breakdown: SkillMatchBreakdown,
    experience_match_score: float,
    role_alignment_score: float,
    trajectory_boost_score: float,
) -> str:
    core_total = len(skill_breakdown.matched_core_skills) + len(skill_breakdown.missing_core_skills)
    secondary_total = len(skill_breakdown.matched_secondary_skills) + len(
        skill_breakdown.missing_secondary_skills
    )
    skill_summary = (
        f"Weighted skill match {skill_breakdown.score * 100:.1f}% "
        f"(mandatory {len(skill_breakdown.matched_core_skills)}/{core_total or 0}, "
        f"nice-to-have {len(skill_breakdown.matched_secondary_skills)}/{secondary_total or 0})"
    )
    if skill_breakdown.matched_skills:
        skill_summary += f". Strong matches: {', '.join(skill_breakdown.matched_skills)}"
    if skill_breakdown.missing_core_skills:
        skill_summary += f". Missing critical skills: {', '.join(skill_breakdown.missing_core_skills)}"
    if skill_breakdown.missing_secondary_skills:
        skill_summary += f". Missing secondary skills: {', '.join(skill_breakdown.missing_secondary_skills)}"

    experience_summary = (
        "Experience fit 100.0% (no minimum experience requirement extracted)"
        if parsed_jd.min_experience_years is None
        else (
            f"Experience fit {experience_match_score * 100:.1f}% "
            f"({candidate.total_experience_years:.1f} yrs vs {parsed_jd.min_experience_years:.1f} yrs required)"
        )
    )
    role_summary = (
        f"Role alignment {role_alignment_score * 100:.1f}% "
        f"({candidate.role_title} vs {parsed_jd.role_title or 'unspecified role'})"
    )
    trajectory_summary = f"Career trajectory boost {trajectory_boost_score * 100:.1f}%."
    final_summary = f"Final Match Score {match_score:.1f}%."
    return f"{skill_summary}. {experience_summary}. {role_summary}. {trajectory_summary} {final_summary}"


def score_candidate_match(
    parsed_jd: ParsedJobDescription,
    candidate: Candidate,
    *,
    skills_weight: float = DEFAULT_SKILLS_WEIGHT,
    experience_weight: float = DEFAULT_EXPERIENCE_WEIGHT,
    role_weight: float = DEFAULT_ROLE_WEIGHT,
    semantic_similarity_score: float | None = None,
    cross_encoder_score: float | None = None,
    skill_graph_service: SkillGraphService | None = None,
) -> CandidateMatchResult:
    skill_graph_service = skill_graph_service or SkillGraphService()
    skill_breakdown = calculate_weighted_skill_match(
        parsed_jd,
        candidate,
        skill_graph_service=skill_graph_service,
    )
    experience_match_score = calculate_experience_match(parsed_jd, candidate)
    role_alignment_score = calculate_role_alignment(parsed_jd, candidate)
    trajectory_boost_score = career_trajectory_boost(candidate)

    total_weight = skills_weight + experience_weight + role_weight
    base_score = (
        (skills_weight * skill_breakdown.score)
        + (experience_weight * experience_match_score)
        + (role_weight * role_alignment_score)
    ) / total_weight
    match_score = min(base_score + trajectory_boost_score, 1.0) * 100

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
        trajectory_boost_score=round(trajectory_boost_score, 4),
        cross_encoder_score=cross_encoder_score,
        matched_skills=skill_breakdown.matched_skills,
        missing_skills=skill_breakdown.missing_skills,
        matched_core_skills=skill_breakdown.matched_core_skills,
        missing_core_skills=skill_breakdown.missing_core_skills,
        matched_secondary_skills=skill_breakdown.matched_secondary_skills,
        missing_secondary_skills=skill_breakdown.missing_secondary_skills,
        skill_alignment_details=skill_breakdown.skill_alignment_details,
        explanation=build_match_explanation(
            candidate,
            parsed_jd,
            match_score,
            skill_breakdown,
            experience_match_score,
            role_alignment_score,
            trajectory_boost_score,
        ),
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
    cross_encoder_lookup: dict[str, float] | None = None,
    skill_graph_service: SkillGraphService | None = None,
) -> list[CandidateMatchResult]:
    similarity_lookup = similarity_lookup or {}
    cross_encoder_lookup = cross_encoder_lookup or {}
    skill_graph_service = skill_graph_service or SkillGraphService()
    results = [
        score_candidate_match(
            parsed_jd,
            candidate,
            skills_weight=skills_weight,
            experience_weight=experience_weight,
            role_weight=role_weight,
            semantic_similarity_score=similarity_lookup.get(candidate.id),
            cross_encoder_score=cross_encoder_lookup.get(candidate.id),
            skill_graph_service=skill_graph_service,
        )
        for candidate in candidates
    ]

    return sorted(
        results,
        key=lambda item: (
            item.cross_encoder_score if item.cross_encoder_score is not None else -1.0,
            item.match_score,
            item.skill_match_score,
            item.semantic_similarity_score if item.semantic_similarity_score is not None else -1.0,
            item.total_experience_years,
        ),
        reverse=True,
    )
