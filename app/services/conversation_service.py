import asyncio
import json
import logging

from app.core.config import get_settings
from app.schemas.candidate import Candidate
from app.schemas.job_description import ParsedJobDescription
from app.schemas.outreach import RecruiterOutreach
from app.services.interest_scoring import CandidateInterestResult, calculate_salary_alignment
from app.services.match_scoring import CandidateMatchResult
from app.services.pii import mask_candidate_payload

logger = logging.getLogger(__name__)


def build_recruiter_outreach_prompt(
    candidate: Candidate,
    parsed_jd: ParsedJobDescription,
    match_result: CandidateMatchResult,
    interest_result: CandidateInterestResult,
) -> dict:
    return {
        "task": "Write a concise recruiter outreach email under 120 words.",
        "candidate": mask_candidate_payload(candidate),
        "role": parsed_jd.model_dump(),
        "signals": {
            "match_score": match_result.match_score,
            "interest_score": interest_result.interest_score,
            "salary_alignment": interest_result.salary_alignment,
            "availability_days": interest_result.availability_days,
            "matched_core_skills": match_result.matched_core_skills,
        },
        "rules": [
            "Do not include placeholders or markdown.",
            "Acknowledge strong fit and one concrete skill area.",
            "Avoid any protected or demographic references.",
            "Do not mention internal scores directly.",
        ],
    }


def build_summary_prompt(
    candidate: Candidate,
    parsed_jd: ParsedJobDescription,
    match_result: CandidateMatchResult,
    interest_result: CandidateInterestResult,
) -> dict:
    return {
        "task": "Write one concise recruiter summary sentence under 35 words.",
        "candidate": mask_candidate_payload(candidate),
        "role": {
            "role_title": parsed_jd.role_title,
            "mandatory_skills": parsed_jd.mandatory_skills,
            "nice_to_have_skills": parsed_jd.nice_to_have_skills,
            "salary_range_usd": parsed_jd.salary_range_usd,
        },
        "signals": {
            "match_score": match_result.match_score,
            "interest_score": interest_result.interest_score,
            "flight_risk_score": interest_result.flight_risk_score,
            "missing_core_skills": match_result.missing_core_skills,
            "salary_alignment": interest_result.salary_alignment,
        },
        "rules": [
            "Be factual and recruiter-friendly.",
            "Mention the strongest fit and the primary risk.",
            "Do not mention protected traits or PII.",
        ],
    }


class BaseCommunicationLLM:
    provider = "base"
    model_name = "base"

    async def generate_text(self, prompt: dict, *, max_tokens: int) -> str:
        raise NotImplementedError


class AsyncGroqCommunicationLLM(BaseCommunicationLLM):
    provider = "groq"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.groq_api_key
        self.model_name = model_name or settings.groq_model
        self.temperature = settings.groq_temperature if temperature is None else temperature
        self.timeout_seconds = settings.llm_timeout_seconds
        self.max_retries = settings.llm_max_retries

    async def generate_text(self, prompt: dict, *, max_tokens: int) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        try:
            from groq import AsyncGroq
        except ImportError as exc:
            raise RuntimeError(
                "groq is not installed for this Python environment. "
                "Use Python 3.11 and run `pip install -r requirements.txt`."
            ) from exc

        client = AsyncGroq(api_key=self.api_key)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a recruiting copilot. Respond with plain text only. "
                    "Keep responses concise, grounded in the provided structured data, "
                    "and avoid protected-class or demographic inferences."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
        ]

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                completion = await asyncio.wait_for(
                    client.chat.completions.create(
                        messages=messages,
                        model=self.model_name,
                        temperature=self.temperature,
                        max_completion_tokens=max_tokens,
                    ),
                    timeout=self.timeout_seconds,
                )
                return (completion.choices[0].message.content or "").strip()
            except Exception as exc:
                last_error = exc
                logger.warning("Groq request failed on attempt %s: %s", attempt, exc)
                if attempt <= self.max_retries:
                    await asyncio.sleep(0.4 * attempt)

        raise RuntimeError(f"Groq request failed after retries: {last_error}") from last_error


class DeterministicCommunicationLLM(BaseCommunicationLLM):
    provider = "deterministic"
    model_name = "rule-based"

    async def generate_text(self, prompt: dict, *, max_tokens: int) -> str:
        task = prompt.get("task", "")
        signals = prompt.get("signals", {})
        role = prompt.get("role", {})

        if "outreach" in task.lower():
            skill = (signals.get("matched_core_skills") or ["your background"])[0]
            role_title = role.get("role_title") or "this role"
            availability = signals.get("availability_days")
            timing = (
                f" You also look available in roughly {availability} days."
                if availability is not None
                else ""
            )
            return (
                f"Hi, I am reaching out about a {role_title} opportunity because your experience in "
                f"{skill} stands out as a strong fit for the role requirements.{timing} "
                "If the scope looks relevant, I would love to share more details."
            )

        missing = signals.get("missing_core_skills") or signals.get("missing_skills") or []
        strongest_fit = (signals.get("matched_core_skills") or ["strong ML/backend alignment"])[0]
        risk = (
            f"missing {missing[0]}"
            if missing
            else f"salary alignment is {signals.get('salary_alignment', 'unknown')}"
        )
        return f"Strong fit in {strongest_fit}, with a watch-out that {risk}."


class RecruiterCommunicationService:
    def __init__(self, llm: BaseCommunicationLLM | None = None) -> None:
        settings = get_settings()
        if llm is not None:
            self.llm = llm
        elif settings.groq_api_key:
            self.llm = AsyncGroqCommunicationLLM()
        else:
            self.llm = DeterministicCommunicationLLM()
        self.fallback_llm = DeterministicCommunicationLLM()

    async def generate_outreach(
        self,
        candidate: Candidate,
        parsed_jd: ParsedJobDescription,
        match_result: CandidateMatchResult,
        interest_result: CandidateInterestResult,
    ) -> RecruiterOutreach:
        prompt = build_recruiter_outreach_prompt(candidate, parsed_jd, match_result, interest_result)
        try:
            message = await self.llm.generate_text(prompt, max_tokens=180)
            return RecruiterOutreach(
                message=message,
                provider=self.llm.provider,
                model=self.llm.model_name,
            )
        except Exception as exc:
            logger.warning("Outreach generation fell back to deterministic mode: %s", exc)
            message = await self.fallback_llm.generate_text(prompt, max_tokens=180)
            return RecruiterOutreach(
                message=message,
                provider=self.fallback_llm.provider,
                model=self.fallback_llm.model_name,
                fallback_reason=str(exc),
            )

    async def generate_summary(
        self,
        candidate: Candidate,
        parsed_jd: ParsedJobDescription,
        match_result: CandidateMatchResult,
        interest_result: CandidateInterestResult,
    ) -> tuple[str, str, str | None]:
        prompt = build_summary_prompt(candidate, parsed_jd, match_result, interest_result)
        try:
            summary = await self.llm.generate_text(prompt, max_tokens=90)
            return summary, self.llm.provider, None
        except Exception as exc:
            logger.warning("Summary generation fell back to deterministic mode: %s", exc)
            summary = await self.fallback_llm.generate_text(prompt, max_tokens=90)
            return summary, self.fallback_llm.provider, str(exc)


__all__ = [
    "AsyncGroqCommunicationLLM",
    "DeterministicCommunicationLLM",
    "RecruiterCommunicationService",
    "build_recruiter_outreach_prompt",
    "build_summary_prompt",
    "calculate_salary_alignment",
]
