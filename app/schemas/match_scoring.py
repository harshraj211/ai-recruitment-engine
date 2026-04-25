from pydantic import BaseModel, Field


class CandidateMatchResult(BaseModel):
    candidate_id: str
    full_name: str
    role_title: str
    total_experience_years: float
    match_score: float = Field(ge=0, le=100)
    skill_match_score: float = Field(ge=0, le=1)
    core_skill_score: float = Field(default=1.0, ge=0, le=1)
    secondary_skill_score: float = Field(default=1.0, ge=0, le=1)
    experience_match_score: float = Field(ge=0, le=1)
    role_alignment_score: float = Field(default=1.0, ge=0, le=1)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    matched_core_skills: list[str] = Field(default_factory=list)
    missing_core_skills: list[str] = Field(default_factory=list)
    matched_secondary_skills: list[str] = Field(default_factory=list)
    missing_secondary_skills: list[str] = Field(default_factory=list)
    explanation: str
    semantic_similarity_score: float | None = None
