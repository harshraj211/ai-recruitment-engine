import asyncio
from collections.abc import Awaitable, Callable

from app.schemas.pipeline import MatchPipelineResult
from app.services.final_ranking import FinalRankingService
from app.services.jd_parser import parse_job_description

ProgressCallback = Callable[[dict], Awaitable[None]]


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
        parsed_jd = await asyncio.to_thread(parse_job_description, job_description)
        ranking_run = await self.final_ranking_service.run_ranking_async(
            parsed_jd,
            top_k_search=top_k_search,
            top_k_final=top_k_final,
            page=page,
            page_size=page_size,
            include_outreach=include_outreach,
            progress_callback=progress_callback,
        )

        return MatchPipelineResult(
            parsed_job_description=parsed_jd,
            rankings=ranking_run.rankings,
            total_candidates_retrieved=ranking_run.total_candidates_retrieved,
            total_candidates_ranked=ranking_run.total_candidates_ranked,
            total_candidates_returned=len(ranking_run.rankings),
            page=ranking_run.page,
            page_size=ranking_run.page_size,
            total_pages=ranking_run.total_pages,
        )

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
