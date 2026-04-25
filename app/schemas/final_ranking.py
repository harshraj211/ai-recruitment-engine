from pydantic import BaseModel, Field

from app.schemas.interest_scoring import CandidateInterestResult
from app.schemas.match_scoring import CandidateMatchResult


class FinalCandidateRanking(BaseModel):
    candidate_id: str
    full_name: str
    role_title: str
    final_score: float = Field(ge=0, le=100)
    rank: int = Field(ge=1)
    match_result: CandidateMatchResult
    interest_result: CandidateInterestResult
    final_explanation: str
    skill_match_reason: str = ""
    experience_match_reason: str = ""
    conversation_insight: str = ""


class FinalRankingRun(BaseModel):
    rankings: list[FinalCandidateRanking] = Field(default_factory=list)
    total_candidates_retrieved: int = Field(ge=0)
    total_candidates_ranked: int = Field(ge=0)
