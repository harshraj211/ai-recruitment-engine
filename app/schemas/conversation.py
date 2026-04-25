from typing import Literal

from pydantic import BaseModel, Field


class ConversationAssessment(BaseModel):
    consent_given: bool
    interest_level: Literal["high", "medium", "low"]
    sentiment: Literal["positive", "neutral", "negative"]
    confidence: Literal["high", "medium", "low"]
    specificity: Literal["high", "medium", "low"]


class ConversationDraft(BaseModel):
    consent_response: str
    interest_response: str
    salary_response: str
    availability_response: str
    summary: str
    assessment: ConversationAssessment


class ConversationTurn(BaseModel):
    stage: Literal["consent", "interest", "salary", "availability"]
    speaker: Literal["recruiter", "candidate"]
    message: str


class ConversationSignals(BaseModel):
    consent_given: bool
    interest_level: Literal["high", "medium", "low"]
    sentiment: Literal["positive", "neutral", "negative"]
    confidence: Literal["high", "medium", "low"]
    specificity: Literal["high", "medium", "low"]
    salary_expectation_usd: int | None = None
    salary_alignment: Literal["aligned", "below_range", "above_range", "unknown"]
    availability_days: int | None = None


class CandidateConversation(BaseModel):
    conversation_id: str
    candidate_id: str
    full_name: str
    role_title: str
    provider: str
    model: str
    created_at: str
    summary: str
    transcript: list[ConversationTurn] = Field(default_factory=list)
    signals: ConversationSignals
    storage_path: str
