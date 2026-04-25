from datetime import date

from pydantic import BaseModel, Field, model_validator


SENIORITY_LADDER = {
    "intern": 0,
    "junior": 1,
    "mid-level": 2,
    "senior": 3,
    "lead": 4,
    "principal": 5,
}


class RoleHistoryEntry(BaseModel):
    company: str
    title: str
    start_date: date
    end_date: date | None = None
    skills: list[str] = Field(default_factory=list)
    location: str | None = None


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
    current_company: str | None = None
    role_history: list[RoleHistoryEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_role_history(self):
        if self.role_history:
            if not self.current_company:
                self.current_company = self.role_history[-1].company
            return self

        years = max(self.total_experience_years, 1.0)
        end_date = date.today()
        start_year = max(end_date.year - int(round(years)), 2005)
        start_date = end_date.replace(year=start_year)
        company = self.current_company or "Confidential Company"
        self.current_company = company
        self.role_history = [
            RoleHistoryEntry(
                company=company,
                title=self.role_title,
                start_date=start_date,
                end_date=None,
                skills=self.skills[:],
                location=self.location,
            )
        ]
        return self

    @property
    def company_names(self) -> list[str]:
        companies = [entry.company for entry in self.role_history if entry.company]
        if self.current_company and self.current_company not in companies:
            companies.append(self.current_company)
        return companies

    @property
    def seniority_rank(self) -> int:
        return SENIORITY_LADDER.get(self.seniority, 2)
