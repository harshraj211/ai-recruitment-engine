from pydantic import BaseModel, Field


class Candidate(BaseModel):
    id: str
    full_name: str
    role_title: str
    seniority: str
    location: str
    total_experience_years: float = Field(ge=0)
    skills: list[str] = Field(default_factory=list, min_length=1)
    preferred_roles: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    education: str
    work_preference: str
    current_status: str
    expected_salary_usd: int = Field(gt=0)
    availability_days: int = Field(ge=0)
    profile_summary: str
