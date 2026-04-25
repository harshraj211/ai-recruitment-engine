import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable

from app.core.config import get_settings
from app.schemas.final_ranking import FinalCandidateRanking, FinalRankingRun
from app.schemas.job_description import ParsedJobDescription
from app.services.candidate_store import build_candidate_search_text, load_candidate_lookup
from app.services.conversation_service import RecruiterCommunicationService
from app.services.cross_encoder_service import CrossEncoderService
from app.services.interest_scoring import PredictiveEngagementService
from app.services.match_scoring import rank_candidates_by_match
from app.services.pipeline_errors import PipelineStageError
from app.services.ranking_consistency import (
    build_availability_insight,
    build_experience_match_reason,
    build_final_explanation,
    build_interest_insight,
    build_recommendation,
    build_salary_alignment_reason,
    build_skill_match_reason,
    calculate_final_score,
    candidate_ranking_sort_key,
)
from app.services.response_validation import ResponseValidationService
from app.services.vector_store import CandidateVectorStore, build_job_search_text

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict], Awaitable[None]]


def min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


class FinalRankingService:
    def __init__(
        self,
        *,
        vector_store: CandidateVectorStore | None = None,
        engagement_service: PredictiveEngagementService | None = None,
        communication_service: RecruiterCommunicationService | None = None,
        cross_encoder_service: CrossEncoderService | None = None,
        response_validation_service: ResponseValidationService | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.vector_store = vector_store or CandidateVectorStore()
        self.engagement_service = engagement_service or PredictiveEngagementService()
        self.communication_service = communication_service or RecruiterCommunicationService()
        self.cross_encoder_service = cross_encoder_service or CrossEncoderService(
            settings.cross_encoder_model_name
        )
        self.response_validation_service = response_validation_service or ResponseValidationService()

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
            started_at = time.perf_counter()
            interest_result = await asyncio.wait_for(
                self.engagement_service.score_candidate_async(candidate, parsed_jd),
                timeout=self.settings.candidate_scoring_timeout_seconds,
            )
            summary_task = asyncio.create_task(
                self.communication_service.generate_summary(
                    candidate,
                    parsed_jd,
                    match_result,
                    interest_result,
                )
            )
            recruiter_outreach_task = None
            engagement_conversation_task = None
            if include_outreach:
                recruiter_outreach_task = asyncio.create_task(
                    self.communication_service.generate_outreach(
                        candidate,
                        parsed_jd,
                        match_result,
                        interest_result,
                    )
                )
                if hasattr(self.communication_service, "generate_simulated_conversation"):
                    engagement_conversation_task = asyncio.create_task(
                        self.communication_service.generate_simulated_conversation(
                            candidate,
                            parsed_jd,
                            match_result,
                            interest_result,
                        )
                    )

            recruiter_outreach = None
            engagement_conversation = None
            if recruiter_outreach_task is not None:
                if engagement_conversation_task is not None:
                    gathered = await asyncio.gather(
                        summary_task,
                        recruiter_outreach_task,
                        engagement_conversation_task,
                    )
                    summary, _, _ = gathered[0]
                    recruiter_outreach = gathered[1]
                    engagement_conversation = gathered[2]
                else:
                    summary_payload, recruiter_outreach = await asyncio.gather(
                        summary_task,
                        recruiter_outreach_task,
                    )
                    summary, _, _ = summary_payload
            else:
                summary, _, _ = await summary_task

            cross_encoder_percent = round((match_result.cross_encoder_score or 0.0) * 100.0, 2)
            final_score = calculate_final_score(
                match_result.match_score,
                interest_result.interest_score,
                cross_encoder_percent,
            )
            ranking = FinalCandidateRanking(
                candidate_id=match_result.candidate_id,
                full_name=match_result.full_name,
                role_title=match_result.role_title,
                candidate_name=match_result.full_name,
                match_score=match_result.match_score,
                interest_score=interest_result.interest_score,
                bm25_score=retrieval_result.bm25_score,
                cross_encoder_score=cross_encoder_percent,
                flight_risk_score=interest_result.flight_risk_score,
                final_score=round(final_score, 2),
                rank=1,
                match_result=match_result,
                interest_result=interest_result,
                summary=summary,
                missing_skills=match_result.missing_skills,
                recommendation=build_recommendation(final_score, match_result, interest_result),
                final_explanation=build_final_explanation(
                    final_score,
                    match_result,
                    interest_result,
                    cross_encoder_percent,
                ),
                skill_match_reason=build_skill_match_reason(match_result),
                experience_match_reason=build_experience_match_reason(match_result, candidate),
                interest_insight=build_interest_insight(interest_result),
                salary_alignment_reason=build_salary_alignment_reason(interest_result),
                availability_insight=build_availability_insight(interest_result),
                recruiter_outreach=recruiter_outreach,
                engagement_conversation=engagement_conversation,
            )
            ranking = await self.response_validation_service.validate_candidate_ranking(
                ranking,
                candidate=candidate,
                parsed_jd=parsed_jd,
            )
            logger.debug(
                "Candidate enrichment candidate=%s duration_ms=%.2f final_score=%.2f",
                candidate.id,
                (time.perf_counter() - started_at) * 1000.0,
                ranking.final_score,
            )
            return ranking

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
        total_started_at = time.perf_counter()
        await self._emit_progress(
            progress_callback,
            {"event": "progress", "stage": "retrieval", "message": "Running hybrid retrieval."},
        )
        retrieval_started_at = time.perf_counter()
        try:
            retrieval_results = await asyncio.wait_for(
                self.vector_store.search_parsed_job_async(parsed_jd, top_k=top_k_search),
                timeout=self.settings.pipeline_stage_timeout_seconds,
            )
        except TimeoutError as exc:
            raise PipelineStageError(
                "retrieval",
                "Hybrid retrieval timed out before candidates were returned.",
                code="retrieval_timeout",
                status_code=504,
            ) from exc
        except Exception as exc:
            raise PipelineStageError(
                "retrieval",
                f"Hybrid retrieval failed: {exc}",
                code="retrieval_failed",
                status_code=503,
            ) from exc
        logger.info(
            "Ranking retrieval completed in %.3fs with %s candidates",
            time.perf_counter() - retrieval_started_at,
            len(retrieval_results),
        )
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
        if not candidates:
            raise PipelineStageError(
                "retrieval",
                "Retrieved candidate ids could not be resolved against the local candidate store.",
                code="candidate_lookup_failed",
                status_code=500,
            )

        await self._emit_progress(
            progress_callback,
            {"event": "progress", "stage": "rerank", "message": "Running cross-encoder re-ranking."},
        )
        rerank_candidates = candidates[: min(self.settings.top_k_rerank, len(candidates))]
        rerank_started_at = time.perf_counter()
        try:
            cross_encoder_lookup = await asyncio.wait_for(
                self._score_cross_encoder(parsed_jd, rerank_candidates),
                timeout=self.settings.pipeline_stage_timeout_seconds,
            )
        except TimeoutError as exc:
            logger.warning("Cross-encoder re-ranking timed out: %s", exc)
            cross_encoder_lookup = {candidate.id: 0.0 for candidate in rerank_candidates}
        logger.info(
            "Ranking re-rank completed in %.3fs for %s candidates",
            time.perf_counter() - rerank_started_at,
            len(rerank_candidates),
        )

        match_started_at = time.perf_counter()
        match_results = rank_candidates_by_match(
            parsed_jd,
            candidates,
            similarity_lookup={
                result.candidate_id: result.semantic_similarity_score
                for result in retrieval_results
            },
            cross_encoder_lookup=cross_encoder_lookup,
        )
        logger.info(
            "Ranking technical scoring completed in %.3fs for %s candidates",
            time.perf_counter() - match_started_at,
            len(match_results),
        )

        await self._emit_progress(
            progress_callback,
            {"event": "progress", "stage": "engagement", "message": "Scoring engagement and summaries."},
        )
        semaphore = asyncio.Semaphore(4)
        enrichment_started_at = time.perf_counter()
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
            try:
                item = await task
            except Exception as exc:
                logger.exception("Candidate enrichment failed: %s", exc)
                continue
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
        logger.info(
            "Ranking enrichment completed in %.3fs with %s validated candidates",
            time.perf_counter() - enrichment_started_at,
            len(ranked_results),
        )
        if not ranked_results:
            raise PipelineStageError(
                "enrichment",
                "Candidate enrichment failed for every shortlisted candidate.",
                code="enrichment_failed",
                status_code=503,
            )

        ranked_results.sort(key=candidate_ranking_sort_key)

        for index, item in enumerate(ranked_results, start=1):
            item.rank = index

        total_pages = max(1, math.ceil(len(ranked_results) / page_size))
        start_index = (page - 1) * page_size
        page_rankings = ranked_results[start_index : start_index + page_size]

        run = FinalRankingRun(
            rankings=page_rankings,
            total_candidates_retrieved=len(retrieval_results),
            total_candidates_ranked=len(match_results),
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
        logger.info(
            "Ranking pipeline completed in %.3fs returning %s candidates on page %s",
            time.perf_counter() - total_started_at,
            len(page_rankings),
            page,
        )
        return self.response_validation_service.validate_ranking_run(run)

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
