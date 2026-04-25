import logging
from collections.abc import Sequence

from app.schemas.final_ranking import FinalCandidateRanking, FinalRankingRun
from app.services.conversation_service import (
    DeterministicCommunicationLLM,
    build_recruiter_outreach_prompt,
    build_summary_prompt,
    outreach_has_data_contradiction,
    summary_has_data_contradiction,
)
from app.services.interest_scoring import (
    calculate_flight_risk_score,
    calculate_interest_score_from_breakdown,
)
from app.services.match_scoring import (
    CORE_SKILL_WEIGHT,
    SECONDARY_SKILL_WEIGHT,
    calculate_match_score_from_components,
    unique_preserving_order,
)
from app.services.ranking_consistency import (
    build_availability_insight,
    build_experience_match_reason,
    build_final_explanation,
    build_interest_insight,
    build_recommendation,
    build_salary_alignment_reason,
    build_skill_match_reason,
    calculate_final_score,
)

logger = logging.getLogger(__name__)


def _word_count(text: str) -> int:
    return len([word for word in text.split() if word.strip()])


def _calculate_skill_coverage(
    matched_core_skills: Sequence[str],
    missing_core_skills: Sequence[str],
    matched_secondary_skills: Sequence[str],
    missing_secondary_skills: Sequence[str],
) -> tuple[float, float, float]:
    core_total = len(matched_core_skills) + len(missing_core_skills)
    secondary_total = len(matched_secondary_skills) + len(missing_secondary_skills)
    core_skill_score = len(matched_core_skills) / core_total if core_total else 1.0
    secondary_skill_score = len(matched_secondary_skills) / secondary_total if secondary_total else 1.0
    weighted_possible = (core_total * CORE_SKILL_WEIGHT) + (secondary_total * SECONDARY_SKILL_WEIGHT)
    weighted_actual = (len(matched_core_skills) * CORE_SKILL_WEIGHT) + (
        len(matched_secondary_skills) * SECONDARY_SKILL_WEIGHT
    )
    skill_match_score = weighted_actual / weighted_possible if weighted_possible else 1.0
    return skill_match_score, core_skill_score, secondary_skill_score


class ResponseValidationService:
    def __init__(self, *, fallback_llm: DeterministicCommunicationLLM | None = None) -> None:
        self.fallback_llm = fallback_llm or DeterministicCommunicationLLM()

    def _sync_match_fields(self, ranking: FinalCandidateRanking, corrections: list[str]) -> None:
        match_result = ranking.match_result
        matched_core_skills = unique_preserving_order(match_result.matched_core_skills)
        missing_core_skills = unique_preserving_order(match_result.missing_core_skills)
        matched_secondary_skills = unique_preserving_order(match_result.matched_secondary_skills)
        missing_secondary_skills = unique_preserving_order(match_result.missing_secondary_skills)
        matched_skills = unique_preserving_order([*matched_core_skills, *matched_secondary_skills])
        missing_skills = unique_preserving_order([*missing_core_skills, *missing_secondary_skills])
        skill_match_score, core_skill_score, secondary_skill_score = _calculate_skill_coverage(
            matched_core_skills,
            missing_core_skills,
            matched_secondary_skills,
            missing_secondary_skills,
        )
        cross_encoder_raw = match_result.cross_encoder_score or 0.0
        if ranking.cross_encoder_score is None:
            cross_encoder_percent = cross_encoder_raw * 100.0
        elif ranking.cross_encoder_score <= 1.0 and cross_encoder_raw == 0.0:
            cross_encoder_percent = ranking.cross_encoder_score * 100.0
        elif ranking.cross_encoder_score <= 1.0 and abs(ranking.cross_encoder_score - cross_encoder_raw) < 0.01:
            cross_encoder_percent = ranking.cross_encoder_score * 100.0
        else:
            cross_encoder_percent = ranking.cross_encoder_score

        if match_result.matched_skills != matched_skills:
            corrections.append("matched_skills")
            match_result.matched_skills = matched_skills
        if match_result.missing_skills != missing_skills:
            corrections.append("missing_skills")
            match_result.missing_skills = missing_skills
        if ranking.missing_skills != missing_skills:
            corrections.append("flat_missing_skills")
            ranking.missing_skills = missing_skills

        match_result.matched_core_skills = matched_core_skills
        match_result.missing_core_skills = missing_core_skills
        match_result.matched_secondary_skills = matched_secondary_skills
        match_result.missing_secondary_skills = missing_secondary_skills
        match_result.skill_match_score = round(skill_match_score, 4)
        match_result.core_skill_score = round(core_skill_score, 4)
        match_result.secondary_skill_score = round(secondary_skill_score, 4)
        match_result.match_score = round(
            calculate_match_score_from_components(
                match_result.skill_match_score,
                match_result.experience_match_score,
                match_result.role_alignment_score,
                match_result.trajectory_boost_score,
                match_result.core_skill_score,
            ),
            2,
        )
        ranking.match_score = match_result.match_score
        ranking.cross_encoder_score = round(cross_encoder_percent, 2)
        match_result.cross_encoder_score = round(ranking.cross_encoder_score / 100.0, 4)

    def _sync_interest_fields(self, ranking: FinalCandidateRanking, corrections: list[str]) -> None:
        interest_result = ranking.interest_result
        interest_result.flight_risk_score = round(
            calculate_flight_risk_score(interest_result.breakdown),
            2,
        )
        interest_result.interest_score = round(
            calculate_interest_score_from_breakdown(interest_result.breakdown),
            2,
        )
        if ranking.interest_score != interest_result.interest_score:
            corrections.append("interest_score")
            ranking.interest_score = interest_result.interest_score
        if ranking.flight_risk_score != interest_result.flight_risk_score:
            corrections.append("flight_risk_score")
            ranking.flight_risk_score = interest_result.flight_risk_score

    async def _validate_summary(
        self,
        ranking: FinalCandidateRanking,
        *,
        candidate,
        parsed_jd,
        corrections: list[str],
    ) -> None:
        if ranking.summary and not summary_has_data_contradiction(
            ranking.summary,
            match_result=ranking.match_result,
            interest_result=ranking.interest_result,
            parsed_jd=parsed_jd,
        ):
            if _word_count(ranking.summary) <= 35:
                return

        corrections.append("summary")
        prompt = build_summary_prompt(
            candidate,
            parsed_jd,
            ranking.match_result,
            ranking.interest_result,
        )
        ranking.summary = await self.fallback_llm.generate_text(prompt, max_tokens=90)

    async def _validate_outreach(
        self,
        ranking: FinalCandidateRanking,
        *,
        candidate,
        parsed_jd,
        corrections: list[str],
    ) -> None:
        if ranking.recruiter_outreach is None:
            return

        if ranking.recruiter_outreach.message and not outreach_has_data_contradiction(
            ranking.recruiter_outreach.message,
            match_result=ranking.match_result,
            interest_result=ranking.interest_result,
            parsed_jd=parsed_jd,
            candidate_role_title=candidate.role_title,
        ):
            if _word_count(ranking.recruiter_outreach.message) <= 120:
                return

        corrections.append("recruiter_outreach")
        prompt = build_recruiter_outreach_prompt(
            candidate,
            parsed_jd,
            ranking.match_result,
            ranking.interest_result,
        )
        ranking.recruiter_outreach.message = await self.fallback_llm.generate_text(prompt, max_tokens=180)
        ranking.recruiter_outreach.provider = self.fallback_llm.provider
        ranking.recruiter_outreach.model = self.fallback_llm.model_name
        ranking.recruiter_outreach.fallback_reason = "Response validation corrected contradictory outreach output."

    async def validate_candidate_ranking(
        self,
        ranking: FinalCandidateRanking,
        *,
        candidate,
        parsed_jd,
    ) -> FinalCandidateRanking:
        corrections: list[str] = []

        ranking.full_name = ranking.match_result.full_name
        ranking.candidate_name = ranking.match_result.full_name
        ranking.role_title = ranking.match_result.role_title

        self._sync_match_fields(ranking, corrections)
        self._sync_interest_fields(ranking, corrections)

        ranking.final_score = round(
            calculate_final_score(
                ranking.match_result.match_score,
                ranking.interest_result.interest_score,
                ranking.cross_encoder_score or 0.0,
            ),
            2,
        )
        ranking.recommendation = build_recommendation(
            ranking.final_score,
            ranking.match_result,
            ranking.interest_result,
        )
        ranking.skill_match_reason = build_skill_match_reason(ranking.match_result)
        ranking.experience_match_reason = build_experience_match_reason(ranking.match_result, candidate)
        ranking.interest_insight = build_interest_insight(ranking.interest_result)
        ranking.salary_alignment_reason = build_salary_alignment_reason(ranking.interest_result)
        ranking.availability_insight = build_availability_insight(ranking.interest_result)
        ranking.final_explanation = build_final_explanation(
            ranking.final_score,
            ranking.match_result,
            ranking.interest_result,
            ranking.cross_encoder_score or 0.0,
        )

        await self._validate_summary(
            ranking,
            candidate=candidate,
            parsed_jd=parsed_jd,
            corrections=corrections,
        )
        await self._validate_outreach(
            ranking,
            candidate=candidate,
            parsed_jd=parsed_jd,
            corrections=corrections,
        )

        if corrections:
            logger.warning(
                "Validated ranking candidate=%s corrected=%s",
                ranking.candidate_id,
                ", ".join(unique_preserving_order(corrections)),
            )

        return ranking

    def validate_ranking_run(self, run: FinalRankingRun) -> FinalRankingRun:
        run.total_candidates_retrieved = max(run.total_candidates_retrieved, 0)
        run.total_candidates_ranked = max(run.total_candidates_ranked, len(run.rankings))
        run.page = max(run.page, 1)
        run.page_size = max(run.page_size, 1)
        run.total_pages = max(run.total_pages, 1)
        return run
