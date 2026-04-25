from pydantic import BaseModel, Field


class InterestScoreBreakdown(BaseModel):
    sentiment_score: float = Field(ge=0, le=1)
    confidence_score: float = Field(ge=0, le=1)
    specificity_score: float = Field(ge=0, le=1)
    salary_match_score: float = Field(ge=0, le=1)
    availability_score: float = Field(ge=0, le=1)


class CandidateInterestResult(BaseModel):
    candidate_id: str
    full_name: str
    role_title: str
    interest_score: float = Field(ge=0, le=100)
    breakdown: InterestScoreBreakdown
    explanation: str
    conversation_id: str
    provider: str
