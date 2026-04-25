"""POST /api/v1/generate-jd – AI-powered job description generator.

Uses Groq LLM to produce a realistic, recruiter-ready job description
for a given role title.  Falls back to a deterministic template when
the LLM is unavailable or fails.
"""

import asyncio
import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response schemas ──────────────────────────────────────

class GenerateJDRequest(BaseModel):
    role: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Target role title, e.g. 'Machine Learning Engineer'.",
        examples=["Machine Learning Engineer", "Backend Engineer", "Data Scientist"],
    )


class GenerateJDResponse(BaseModel):
    job_description: str
    provider: str = "groq"


# ── Deterministic fallback templates ────────────────────────────────

FALLBACK_TEMPLATES: dict[str, str] = {
    "machine learning engineer": (
        "We are hiring a Senior Machine Learning Engineer to design, build, and "
        "deploy production ML systems. The ideal candidate has 4+ years of "
        "experience with Python, PyTorch or TensorFlow, Docker, and cloud "
        "platforms (AWS/GCP). Must be skilled in feature engineering, model "
        "evaluation, and CI/CD for ML pipelines. Nice to have: MLflow, Ray, "
        "vector databases, and RAG architectures. Salary range: $130,000–$175,000. "
        "Remote-friendly."
    ),
    "backend engineer": (
        "We are looking for a Backend Engineer to build and maintain scalable "
        "APIs and microservices. Requirements: 3+ years with Python or Go, "
        "FastAPI or Django, PostgreSQL, Redis, Docker, and Kubernetes. "
        "Experience with event-driven architectures and CI/CD pipelines is "
        "essential. Nice to have: GraphQL, gRPC, and observability tooling "
        "(Prometheus, Grafana). Salary range: $120,000–$160,000. Hybrid."
    ),
    "data scientist": (
        "We are hiring a Data Scientist to drive insights and predictive models "
        "across our product analytics stack. Must have 3+ years of experience "
        "with Python, SQL, Pandas, scikit-learn, and statistical modeling. "
        "Experience with A/B testing, causal inference, and dashboarding tools "
        "(Looker, Tableau) is expected. Nice to have: deep learning, NLP, and "
        "Spark. Salary range: $115,000–$155,000. Remote."
    ),
    "frontend engineer": (
        "We are seeking a Frontend Engineer to craft performant, accessible "
        "web experiences. Requirements: 3+ years with React or Vue, TypeScript, "
        "HTML/CSS, and modern build tools (Vite, Webpack). Must understand "
        "responsive design, accessibility standards, and state management. "
        "Nice to have: Next.js, Storybook, and design-system experience. "
        "Salary range: $110,000–$150,000. Remote-friendly."
    ),
    "devops engineer": (
        "We are hiring a DevOps Engineer to own our cloud infrastructure and "
        "CI/CD pipelines. Requirements: 3+ years with AWS or GCP, Terraform, "
        "Docker, Kubernetes, and GitHub Actions or Jenkins. Must be comfortable "
        "with monitoring (Datadog, Prometheus) and incident response. Nice to "
        "have: service mesh, GitOps, and cost-optimization experience. "
        "Salary range: $125,000–$170,000. Remote."
    ),
}

DEFAULT_FALLBACK = (
    "We are hiring a {role} with strong technical skills and "
    "industry experience. The ideal candidate has 3+ years in the domain, "
    "excellent problem-solving abilities, and experience with modern tooling "
    "and best practices. Competitive salary offered based on experience. "
    "Remote-friendly position."
)


def _get_fallback_jd(role: str) -> str:
    """Return a deterministic JD for a role, using closest match or generic."""
    normalized = role.strip().lower()
    for key, template in FALLBACK_TEMPLATES.items():
        if key in normalized or normalized in key:
            return template
    return DEFAULT_FALLBACK.format(role=role)


# ── Groq prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a senior technical recruiter writing realistic, detailed job "
    "descriptions. Output ONLY the job description text—no markdown, no "
    "headings, no bullet formatting. Write in clear professional prose "
    "paragraphs. Keep the description between 80 and 180 words."
)


def _build_user_prompt(role: str) -> str:
    return (
        f"Generate a realistic job description for a {role}. "
        "Include: required technical skills, years of experience needed, "
        "nice-to-have skills, domain knowledge expectations, a salary range "
        "in USD, and whether the role is remote, hybrid, or on-site. "
        "Write in second-person professional tone addressed to potential "
        "candidates. Do not use markdown, bullet points, or numbered lists."
    )


# ── Endpoint ────────────────────────────────────────────────────────

@router.post(
    "/generate-jd",
    response_model=GenerateJDResponse,
    summary="Generate a job description with AI",
)
@router.post("/generate-jd/", response_model=GenerateJDResponse, include_in_schema=False)
async def generate_job_description(payload: GenerateJDRequest) -> GenerateJDResponse:
    settings = get_settings()

    if not settings.groq_api_key:
        logger.info("No GROQ_API_KEY configured – returning fallback JD for '%s'", payload.role)
        return GenerateJDResponse(
            job_description=_get_fallback_jd(payload.role),
            provider="fallback",
        )

    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=settings.groq_api_key)
        completion = await asyncio.wait_for(
            client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(payload.role)},
                ],
                model=settings.groq_model,
                temperature=0.4,
                max_completion_tokens=350,
            ),
            timeout=settings.llm_timeout_seconds,
        )
        text = (completion.choices[0].message.content or "").strip()
        if len(text) < 40:
            raise ValueError("Groq returned a suspiciously short response.")
        return GenerateJDResponse(job_description=text, provider="groq")

    except Exception as exc:
        logger.warning("Groq JD generation failed, falling back: %s", exc)
        return GenerateJDResponse(
            job_description=_get_fallback_jd(payload.role),
            provider="fallback",
        )
