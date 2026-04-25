import asyncio
import logging
import math
from collections.abc import Awaitable, Callable

from app.core.config import get_settings
from app.schemas.final_ranking import FinalCandidateRanking, FinalRankingRun
from app.schemas.job_description import ParsedJobDescription
from app.services.candidate_store import build_candidate_search_text, load_candidate_lookup
from app.services.conversation_service import RecruiterCommunicationService
from app.services.cross_encoder_service import CrossEncoderService
from app.services.interest_scoring import PredictiveEngagementService
from app.services.match_scoring import apply_mandatory_skill_penalty, rank_candidates_by_match
from app.services.vector_store import CandidateVectorStore, build_job_search_text

logger = logging.getLogger(__name__)

DEFAULT_MATCH_WEIGHT = 0.50
DEFAULT_INTEREST_WEIGHT = 0.25
DEFAULT_CROSS_ENCODER_WEIGHT = 0.25

ProgressCallback = Callable[[dict], Awaitable[None]]


def min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def build_final_explanation(
    final_score: float,
    match_result,
    interest_result,
    cross_encoder_score: float | None,
) -> str:
    return (
        f"Final Score {final_score:.1f}% combines Match {match_result.match_score:.1f}%, "
        f"Interest {interest_result.interest_score:.1f}%, and "
        f"Cross-Encoder {(cross_encoder_score or 0.0) * 100:.1f}%. "
        f"{match_result.explanation} {interest_result.explanation}"
    )


def build_skill_match_reason(match_result) -> str:
    details = "; ".join(match_result.skill_alignment_details[:4])
    return (
        f"Mandatory skill fit {match_result.core_skill_score * 100:.0f}%, "
        f"nice-to-have fit {match_result.secondary_skill_score * 100:.0f}%. "
        f"{details}"
    )


def build_experience_match_reason(match_result, candidate) -> str:
    return (
        f"Experience fit {match_result.experience_match_score * 100:.0f}% with "
        f"{candidate.total_experience_years:.1f} years in {candidate.role_title}. "
        f"Trajectory boost {match_result.trajectory_boost_score * 100:.1f}%."
    )


def build_interest_insight(interest_result) -> str:
    return (
        f"Flight risk {interest_result.flight_risk_score:.1f}% and predicted interest "
        f"{interest_result.interest_score:.1f}%."
    )


def build_salary_alignment_reason(interest_result) -> str:
    return f"Salary alignment is {interest_result.salary_alignment}."


def build_availability_insight(interest_result) -> str:
    availability = interest_result.availability_days
    return f"Availability is {availability} days." if availability is not None else "Availability is unknown."


def build_recommendation(final_score: float, match_result, interest_result) -> str:
    if match_result.missing_core_skills:
        return "Review manually before outreach due to missing critical skills."
    if final_score >= 80 and interest_result.flight_risk_score >= 55:
        return "Prioritize outreach this week."
    if final_score >= 70:
        return "Advance to recruiter screen."
    if final_score >= 60:
        return "Keep warm for secondary review."
    return "Do not prioritize for this requisition."


class FinalRankingService:
    def __init__(
        self,
        *,
        vector_store: CandidateVectorStore | None = None,
        engagement_service: PredictiveEngagementService | None = None,
        communication_service: RecruiterCommunicationService | None = None,
        cross_encoder_service: CrossEncoderService | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.vector_store = vector_store or CandidateVectorStore()
        self.engagement_service = engagement_service or PredictiveEngagementService()
        self.communication_service = communication_service or RecruiterCommunicationService()
        self.cross_encoder_service = cross_encoder_service or CrossEncoderService(
            settings.cross_encoder_model_name
        )

    async def _emit_progress(self, callback: ProgressCallback | None, payload: dict) -> None:
        if callback is not None:
            await callback(payload)

    async def _score_cross_encoder(
        self,
        parsed_jd: ParsedJobDescription,
        candidates,
    ) -> dict[str, float]:
        if not candidates:
            return {}

        pairs = [
            (
                build_job_search_text(parsed_jd),
                build_candidate_search_text(candidate),
            )
            for candidate in candidates
        ]

        try:
            raw_scores = await self.cross_encoder_service.score_pairs_async(pairs)
        except Exception as exc:
            logger.warning("Cross-encoder scoring failed: %s", exc)
            return {candidate.id: 0.0 for candidate in candidates}

        normalized = min_max_normalize(raw_scores)
        return {
            candidate.id: score
            for candidate, score in zip(candidates, normalized, strict=True)
        }

    async def _enrich_candidate(
        self,
        candidate,
        parsed_jd: ParsedJobDescription,
        match_result,
        retrieval_result,
        *,
        include_outreach: bool,
        semaphore: asyncio.Semaphore,
    ) -> FinalCandidateRanking:
        async with semaphore:
            interest_result = await asyncio.wait_for(
                self.engagement_service.score_candidate_async(candidate, parsed_jd),
                timeout=self.settings.candidate_scoring_timeout_seconds,
            )
            summary, _, _ = await self.communication_service.generate_summary(
                candidate,
                parsed_jd,
                match_result,
                interest_result,
            )
            recruiter_outreach = None
            if include_outreach:
                recruiter_outreach = await self.communication_service.generate_outreach(
                    candidate,
                    parsed_jd,
                    match_result,
                    interest_result,
                )

            cross_encoder_score = match_result.cross_encoder_score or 0.0
            final_score = (
                (DEFAULT_MATCH_WEIGHT * match_result.match_score)
                + (DEFAULT_INTEREST_WEIGHT * interest_result.interest_score)
                + (DEFAULT_CROSS_ENCODER_WEIGHT * (cross_encoder_score * 100.0))
            )
            final_score = apply_mandatory_skill_penalty(final_score, match_result.core_skill_score)

            return FinalCandidateRanking(
                candidate_id=match_result.candidate_id,
                full_name=match_result.full_name,
                role_title=match_result.role_title,
                candidate_name=match_result.full_name,
                match_score=match_result.match_score,
                interest_score=interest_result.interest_score,
                bm25_score=retrieval_result.bm25_score,
                cross_encoder_score=round(cross_encoder_score * 100.0, 2),
                flight_risk_score=interest_result.flight_risk_score,
                final_score=round(final_score, 2),
                rank=1,
                match_result=match_result,
                interest_result=interest_result,
                summary=summary,
                missing_skills=match_result.missing_core_skills or match_result.missing_skills,
                recommendation=build_recommendation(final_score, match_result, interest_result),
                final_explanation=build_final_explanation(
                    final_score,
                    match_result,
                    interest_result,
                    cross_encoder_score,
                ),
                skill_match_reason=build_skill_match_reason(match_result),
                experience_match_reason=build_experience_match_reason(match_result, candidate),
                interest_insight=build_interest_insight(interest_result),
                salary_alignment_reason=build_salary_alignment_reason(interest_result),
                availability_insight=build_availability_insight(interest_result),
                recruiter_outreach=recruiter_outreach,
            )

    async def rank_candidates_async(
        self,
        parsed_jd: ParsedJobDescription,
        *,
        top_k_search: int = 10,
        top_k_final: int = 5,
        page: int = 1,
        page_size: int = 5,
        include_outreach: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> list[FinalCandidateRanking]:
        run = await self.run_ranking_async(
            parsed_jd,
            top_k_search=top_k_search,
            top_k_final=top_k_final,
            page=page,
            page_size=page_size,
            include_outreach=include_outreach,
            progress_callback=progress_callback,
        )
        return run.rankings

    async def run_ranking_async(
        self,
        parsed_jd: ParsedJobDescription,
        *,
        top_k_search: int = 10,
        top_k_final: int = 5,
        page: int = 1,
        page_size: int = 5,
        include_outreach: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> FinalRankingRun:
        await self._emit_progress(
            progress_callback,
            {"event": "progress", "stage": "retrieval", "message": "Running hybrid retrieval."},
        )
        retrieval_results = await self.vector_store.search_parsed_job_async(parsed_jd, top_k=top_k_search)
        if not retrieval_results:
            return FinalRankingRun(
                rankings=[],
                total_candidates_retrieved=0,
                total_candidates_ranked=0,
                page=page,
                page_size=page_size,
                total_pages=1,
            )

        candidate_lookup = load_candidate_lookup()
        candidates = [
            candidate_lookup[result.candidate_id]
            for result in retrieval_results
            if result.candidate_id in candidate_lookup
        ]
        retrieval_lookup = {result.candidate_id: result for result in retrieval_results}

        await self._emit_progress(
            progress_callback,
            {"event": "progress", "stage": "rerank", "message": "Running cross-encoder re-ranking."},
        )
        rerank_candidates = candidates[: min(self.settings.top_k_rerank, len(candidates))]
        cross_encoder_lookup = await self._score_cross_encoder(parsed_jd, rerank_candidates)

        match_results = rank_candidates_by_match(
            parsed_jd,
            candidates,
            similarity_lookup={
                result.candidate_id: result.semantic_similarity_score
                for result in retrieval_results
            },
            cross_encoder_lookup=cross_encoder_lookup,
        )

        await self._emit_progress(
            progress_callback,
            {"event": "progress", "stage": "engagement", "message": "Scoring engagement and summaries."},
        )
        semaphore = asyncio.Semaphore(4)
        tasks = [
            self._enrich_candidate(
                candidate_lookup[match_result.candidate_id],
                parsed_jd,
                match_result,
                retrieval_lookup[match_result.candidate_id],
                include_outreach=include_outreach,
                semaphore=semaphore,
            )
            for match_result in match_results[: max(top_k_final, page * page_size)]
        ]
        ranked_results = []
        for task in asyncio.as_completed(tasks):
            item = await task
            if item is None:
                continue
            ranked_results.append(item)
            await self._emit_progress(
                progress_callback,
                {
                    "event": "candidate",
                    "payload": item.model_dump(),
                },
            )

        ranked_results.sort(
            key=lambda item: (
                item.final_score,
                item.cross_encoder_score or 0.0,
                item.match_result.match_score,
                item.interest_result.interest_score,
            ),
            reverse=True,
        )

        for index, item in enumerate(ranked_results, start=1):
            item.rank = index

        total_pages = max(1, math.ceil(len(ranked_results) / page_size))
        start_index = (page - 1) * page_size
        page_rankings = ranked_results[start_index : start_index + page_size]

        return FinalRankingRun(
            rankings=page_rankings,
            total_candidates_retrieved=len(retrieval_results),
            total_candidates_ranked=len(match_results),
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    def rank_candidates(
        self,
        parsed_jd: ParsedJobDescription,
        *,
        top_k_search: int = 10,
        top_k_final: int = 5,
        page: int = 1,
        page_size: int = 5,
        include_outreach: bool = False,
    ) -> list[FinalCandidateRanking]:
        return asyncio.run(
            self.rank_candidates_async(
                parsed_jd,
                top_k_search=top_k_search,
                top_k_final=top_k_final,
                page=page,
                page_size=page_size,
                include_outreach=include_outreach,
            )
        )

    def run_ranking(
        self,
        parsed_jd: ParsedJobDescription,
        *,
        top_k_search: int = 10,
        top_k_final: int = 5,
        page: int = 1,
        page_size: int = 5,
        include_outreach: bool = False,
    ) -> FinalRankingRun:
        return asyncio.run(
            self.run_ranking_async(
                parsed_jd,
                top_k_search=top_k_search,
                top_k_final=top_k_final,
                page=page,
                page_size=page_size,
                include_outreach=include_outreach,
            )
        )
