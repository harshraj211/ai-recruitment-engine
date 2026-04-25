from pydantic import BaseModel, Field


class ParsedJobDescription(BaseModel):
    raw_text: str
    role_title: str | None = None
    seniority: str | None = None
    min_experience_years: float | None = Field(default=None, ge=0)
    skills: list[str] = Field(default_factory=list)
    mandatory_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    domain_knowledge: list[str] = Field(default_factory=list)
    core_skills: list[str] = Field(default_factory=list)
    secondary_skills: list[str] = Field(default_factory=list)
    salary_range_usd: list[int] = Field(default_factory=list)
    work_mode: str | None = None
