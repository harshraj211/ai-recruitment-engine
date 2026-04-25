from typing import Literal

from pydantic import BaseModel, Field


class InterestScoreBreakdown(BaseModel):
    salary_alignment_score: float = Field(ge=0, le=1)
    availability_score: float = Field(ge=0, le=1)
    role_relevance_score: float = Field(ge=0, le=1)
    engagement_probability_score: float = Field(ge=0, le=1)


class CandidateInterestResult(BaseModel):
    candidate_id: str
    full_name: str
    role_title: str
    interest_score: float = Field(ge=0, le=100)
    breakdown: InterestScoreBreakdown
    salary_alignment: Literal["aligned", "below_range", "above_range", "unknown"]
    availability_days: int | None = None
    model_version: str = "predicted-interest-v1"
    explanation: str
    provider: str = "deterministic"
