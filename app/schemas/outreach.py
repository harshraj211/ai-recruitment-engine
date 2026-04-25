from pydantic import BaseModel


class RecruiterOutreach(BaseModel):
    message: str
    provider: str
    model: str | None = None
    fallback_reason: str | None = None
