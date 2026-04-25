from pydantic import BaseModel, Field

from app.schemas.final_ranking import FinalCandidateRanking
from app.schemas.job_description import ParsedJobDescription


class MatchPipelineResult(BaseModel):
    parsed_job_description: ParsedJobDescription
    rankings: list[FinalCandidateRanking] = Field(default_factory=list)
    total_candidates_retrieved: int = Field(ge=0)
    total_candidates_ranked: int = Field(ge=0)
    total_candidates_returned: int = Field(ge=0)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=5, ge=1)
    total_pages: int = Field(default=1, ge=1)
