from pydantic import BaseModel, Field

from app.schemas.interest_scoring import CandidateInterestResult
from app.schemas.match_scoring import CandidateMatchResult
from app.schemas.outreach import RecruiterOutreach


class FinalCandidateRanking(BaseModel):
    candidate_id: str
    full_name: str
    role_title: str
    final_score: float = Field(ge=0, le=100)
    rank: int = Field(ge=1)
    match_result: CandidateMatchResult
    interest_result: CandidateInterestResult
    candidate_name: str | None = None
    match_score: float | None = None
    interest_score: float | None = None
    bm25_score: float | None = None
    cross_encoder_score: float | None = None
    flight_risk_score: float | None = None
    summary: str = ""
    missing_skills: list[str] = Field(default_factory=list)
    recommendation: str = ""
    final_explanation: str
    skill_match_reason: str = ""
    experience_match_reason: str = ""
    interest_insight: str = ""
    salary_alignment_reason: str = ""
    availability_insight: str = ""
    recruiter_outreach: RecruiterOutreach | None = None


class FinalRankingRun(BaseModel):
    rankings: list[FinalCandidateRanking] = Field(default_factory=list)
    total_candidates_retrieved: int = Field(ge=0)
    total_candidates_ranked: int = Field(ge=0)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=5, ge=1)
    total_pages: int = Field(default=1, ge=1)
