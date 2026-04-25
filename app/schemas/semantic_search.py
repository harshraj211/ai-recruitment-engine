from pydantic import BaseModel, Field


class SemanticSearchResult(BaseModel):
    candidate_id: str
    full_name: str
    role_title: str
    total_experience_years: float
    skills: list[str] = Field(default_factory=list)
    similarity_score: float
    semantic_similarity_score: float | None = None
    keyword_match_score: float | None = None
    bm25_score: float | None = None
    dense_profile_score: float | None = None
    dense_skill_score: float | None = None
    rrf_score: float | None = None
    profile_summary: str
