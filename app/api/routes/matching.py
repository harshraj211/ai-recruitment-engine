from fastapi import APIRouter, Depends, HTTPException

from app.schemas.api import MatchRequest, MatchResponse
from app.services.pipeline_service import MatchPipelineService

router = APIRouter()


def get_match_pipeline_service() -> MatchPipelineService:
    return MatchPipelineService()


@router.post("/match", response_model=MatchResponse)
def match_candidates(
    payload: MatchRequest,
    pipeline_service: MatchPipelineService = Depends(get_match_pipeline_service),
) -> MatchResponse:
    try:
        result = pipeline_service.run(
            payload.job_description,
            top_k_search=payload.top_k_search,
            top_k_final=payload.top_k_final,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return MatchResponse(
        parsed_job_description=result.parsed_job_description,
        rankings=result.rankings,
        total_candidates_considered=result.total_candidates_ranked,
        total_candidates_retrieved=result.total_candidates_retrieved,
        total_candidates_returned=result.total_candidates_returned,
    )
