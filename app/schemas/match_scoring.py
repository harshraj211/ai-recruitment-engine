from pydantic import BaseModel, Field


class CandidateMatchResult(BaseModel):
    candidate_id: str
    full_name: str
    role_title: str
    total_experience_years: float
    match_score: float = Field(ge=0, le=100)
    skill_match_score: float = Field(ge=0, le=1)
    experience_match_score: float = Field(ge=0, le=1)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    explanation: str
    semantic_similarity_score: float | None = None
