from app.schemas.pipeline import MatchPipelineResult
from app.services.final_ranking import FinalRankingService
from app.services.jd_parser import parse_job_description


class MatchPipelineService:
    """Coordinates the end-to-end JD -> shortlist pipeline."""

    def __init__(
        self,
        *,
        final_ranking_service: FinalRankingService | None = None,
    ) -> None:
        self.final_ranking_service = final_ranking_service or FinalRankingService()

    def run(
        self,
        job_description: str,
        *,
        top_k_search: int = 5,
        top_k_final: int = 5,
    ) -> MatchPipelineResult:
        parsed_jd = parse_job_description(job_description)
        ranking_run = self.final_ranking_service.run_ranking(
            parsed_jd,
            top_k_search=top_k_search,
            top_k_final=top_k_final,
        )

        return MatchPipelineResult(
            parsed_job_description=parsed_jd,
            rankings=ranking_run.rankings,
            total_candidates_retrieved=ranking_run.total_candidates_retrieved,
            total_candidates_ranked=ranking_run.total_candidates_ranked,
            total_candidates_returned=len(ranking_run.rankings),
        )
