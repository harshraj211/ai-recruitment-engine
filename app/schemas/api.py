from pydantic import BaseModel, Field

from app.schemas.final_ranking import FinalCandidateRanking
from app.schemas.job_description import ParsedJobDescription


class MatchRequest(BaseModel):
    job_description: str = Field(min_length=20)
    top_k_search: int = Field(default=5, ge=1, le=20)
    top_k_final: int = Field(default=5, ge=1, le=10)
    include_outreach: bool = False


class MatchResponse(BaseModel):
    parsed_job_description: ParsedJobDescription
    rankings: list[FinalCandidateRanking]
    total_candidates_considered: int = Field(ge=0)
    total_candidates_retrieved: int = Field(ge=0)
    total_candidates_returned: int = Field(ge=0)
