from pydantic import BaseModel, Field


class SemanticSearchResult(BaseModel):
    candidate_id: str
    full_name: str
    role_title: str
    total_experience_years: float
    skills: list[str] = Field(default_factory=list)
    similarity_score: float
    profile_summary: str
