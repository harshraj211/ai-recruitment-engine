import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from app.schemas.pipeline import MatchPipelineResult
from app.services.final_ranking import FinalRankingService
from app.services.jd_parser import parse_job_description
from app.services.pipeline_errors import PipelineStageError

ProgressCallback = Callable[[dict], Awaitable[None]]

logger = logging.getLogger(__name__)


class MatchPipelineService:
    """Coordinates the end-to-end JD -> shortlist pipeline."""

    def __init__(
        self,
        *,
        final_ranking_service: FinalRankingService | None = None,
    ) -> None:
        self.final_ranking_service = final_ranking_service or FinalRankingService()

    async def run_async(
        self,
        job_description: str,
        *,
        top_k_search: int = 10,
        top_k_final: int = 5,
        page: int = 1,
        page_size: int = 5,
        include_outreach: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> MatchPipelineResult:
        total_started_at = time.perf_counter()
        parse_started_at = time.perf_counter()
        try:
            parsed_jd = await asyncio.wait_for(
                asyncio.to_thread(parse_job_description, job_description),
                timeout=self.final_ranking_service.settings.pipeline_stage_timeout_seconds,
            )
        except TimeoutError as exc:
            raise PipelineStageError(
                "parse",
                "Job description parsing timed out.",
                code="parse_timeout",
                status_code=504,
            ) from exc
        except Exception as exc:
            raise PipelineStageError(
                "parse",
                f"Job description parsing failed: {exc}",
                code="parse_failed",
                status_code=422,
            ) from exc
        logger.info("Pipeline parse completed in %.3fs", time.perf_counter() - parse_started_at)

        try:
            ranking_run = await self.final_ranking_service.run_ranking_async(
                parsed_jd,
                top_k_search=top_k_search,
                top_k_final=top_k_final,
                page=page,
                page_size=page_size,
                include_outreach=include_outreach,
                progress_callback=progress_callback,
            )
        except PipelineStageError:
            raise
        except Exception as exc:
            raise PipelineStageError(
                "ranking",
                f"Candidate ranking failed: {exc}",
                code="ranking_failed",
                status_code=503,
            ) from exc

        result = MatchPipelineResult(
            parsed_job_description=parsed_jd,
            rankings=ranking_run.rankings,
            total_candidates_retrieved=ranking_run.total_candidates_retrieved,
            total_candidates_ranked=ranking_run.total_candidates_ranked,
            total_candidates_returned=len(ranking_run.rankings),
            page=ranking_run.page,
            page_size=ranking_run.page_size,
            total_pages=ranking_run.total_pages,
        )
        logger.info(
            "Pipeline completed in %.3fs returning %s ranked candidates",
            time.perf_counter() - total_started_at,
            result.total_candidates_returned,
        )
        return result

    def run(
        self,
        job_description: str,
        *,
        top_k_search: int = 10,
        top_k_final: int = 5,
        page: int = 1,
        page_size: int = 5,
        include_outreach: bool = False,
    ) -> MatchPipelineResult:
        return asyncio.run(
            self.run_async(
                job_description,
                top_k_search=top_k_search,
                top_k_final=top_k_final,
                page=page,
                page_size=page_size,
                include_outreach=include_outreach,
            )
        )
