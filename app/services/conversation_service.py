import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings
from app.schemas.candidate import Candidate
from app.schemas.conversation import (
    CandidateConversation,
    ConversationAssessment,
    ConversationDraft,
    ConversationSignals,
    ConversationTurn,
)
from app.schemas.job_description import ParsedJobDescription
from app.services.match_scoring import calculate_skill_match


def calculate_salary_alignment(
    salary_expectation_usd: int | None,
    salary_range_usd: list[int],
) -> str:
    if salary_expectation_usd is None or len(salary_range_usd) != 2:
        return "unknown"

    low, high = sorted(salary_range_usd)
    if low <= salary_expectation_usd <= high:
        return "aligned"
    if salary_expectation_usd < low:
        return "below_range"
    return "above_range"


def build_recruiter_prompts(
    parsed_jd: ParsedJobDescription,
    recruiter_name: str,
) -> dict[str, str]:
    role_title = parsed_jd.role_title or "this role"

    salary_prompt = (
        f"Our budget for this role is ${parsed_jd.salary_range_usd[0]:,} to "
        f"${parsed_jd.salary_range_usd[1]:,} USD. Does that align with your expectations?"
        if len(parsed_jd.salary_range_usd) == 2
        else "What compensation range are you targeting for your next move?"
    )

    return {
        "consent": (
            f"Hi, I'm {recruiter_name}. I'm reaching out about a {role_title} opportunity. "
            "Do you have a couple of minutes to chat?"
        ),
        "interest": (
            f"What about this {role_title} role sounds interesting to you, "
            "and how closely does it match your recent work?"
        ),
        "salary": salary_prompt,
        "availability": "If there is mutual interest, when would you be available to start?",
    }


class BaseConversationLLM:
    provider = "base"
    model_name = "base"

    def generate_draft(
        self,
        candidate: Candidate,
        parsed_jd: ParsedJobDescription,
        recruiter_prompts: dict[str, str],
    ) -> ConversationDraft:
        raise NotImplementedError


class GroqConversationLLM(BaseConversationLLM):
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

    def _build_messages(
        self,
        candidate: Candidate,
        parsed_jd: ParsedJobDescription,
        recruiter_prompts: dict[str, str],
    ) -> list[dict[str, str]]:
        system_prompt = """
You are simulating a job candidate in a recruiting demo.
Respond only with valid JSON.
Do not include markdown or extra commentary.
Use the candidate profile as the source of truth.
Keep each candidate response natural, concise, and realistic.
Return exactly this JSON shape:
{
  "consent_response": "string",
  "interest_response": "string",
  "salary_response": "string",
  "availability_response": "string",
  "summary": "string",
  "assessment": {
    "consent_given": true,
    "interest_level": "high|medium|low",
    "sentiment": "positive|neutral|negative",
    "confidence": "high|medium|low",
    "specificity": "high|medium|low"
  }
}
""".strip()

        user_prompt = json.dumps(
            {
                "task": "Generate candidate-side replies for a four-stage recruiter conversation.",
                "candidate_profile": candidate.model_dump(),
                "parsed_job_description": parsed_jd.model_dump(),
                "recruiter_prompts": recruiter_prompts,
                "rules": [
                    "Candidate responses should fit the profile and current status.",
                    "Mention concrete matching skills when appropriate.",
                    "Salary response should reflect the candidate's expected salary.",
                    "Availability response should reflect the candidate's availability_days.",
                    "Summary should be one short paragraph.",
                ],
            },
            indent=2,
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_draft(
        self,
        candidate: Candidate,
        parsed_jd: ParsedJobDescription,
        recruiter_prompts: dict[str, str],
    ) -> ConversationDraft:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError(
                "groq is not installed for this Python environment. "
                "Use Python 3.11 and run `pip install -r requirements.txt`."
            ) from exc

        client = Groq(api_key=self.api_key)
        completion = client.chat.completions.create(
            messages=self._build_messages(candidate, parsed_jd, recruiter_prompts),
            model=self.model_name,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            max_completion_tokens=900,
            seed=7,
        )

        content = completion.choices[0].message.content or "{}"
        try:
            return ConversationDraft.model_validate_json(content)
        except Exception as exc:
            raise RuntimeError(f"Groq returned invalid conversation JSON: {content}") from exc


class MockConversationLLM(BaseConversationLLM):
    provider = "mock"
    model_name = "mock-local"

    def generate_draft(
        self,
        candidate: Candidate,
        parsed_jd: ParsedJobDescription,
        recruiter_prompts: dict[str, str],
    ) -> ConversationDraft:
        skill_match_score, matched_skills, missing_skills = calculate_skill_match(parsed_jd, candidate)
        salary_alignment = calculate_salary_alignment(
            candidate.expected_salary_usd,
            parsed_jd.salary_range_usd,
        )

        if candidate.current_status == "open_to_work" and skill_match_score >= 0.5:
            interest_level = "high"
            sentiment = "positive"
            confidence = "high"
        elif candidate.current_status in {"exploring", "passive"} and skill_match_score >= 0.3:
            interest_level = "medium"
            sentiment = "neutral"
            confidence = "medium"
        else:
            interest_level = "low"
            sentiment = "negative"
            confidence = "low"

        specificity = "high" if len(matched_skills) >= 2 else "medium"
        consent_given = interest_level != "low"

        if matched_skills:
            skills_text = ", ".join(matched_skills[:3])
            interest_response = (
                f"Yes, I'm interested because the role lines up well with my recent work in {skills_text}. "
                f"My background as a {candidate.role_title} feels relevant here."
            )
        else:
            interest_response = (
                "I would need more detail to judge the fit because the role overlaps only partially "
                "with my recent work."
            )

        if salary_alignment == "aligned":
            salary_response = (
                f"My target is around ${candidate.expected_salary_usd:,} USD, "
                "so your range looks workable."
            )
        elif salary_alignment == "above_range":
            salary_response = (
                f"I'm targeting about ${candidate.expected_salary_usd:,} USD, "
                "which is a bit above the shared range, though I could discuss total scope."
            )
        elif salary_alignment == "below_range":
            salary_response = (
                f"I'm looking for around ${candidate.expected_salary_usd:,} USD, "
                "so the budget is comfortably within range for me."
            )
        else:
            salary_response = f"I'm targeting around ${candidate.expected_salary_usd:,} USD."

        availability_response = (
            f"I would likely be able to start in about {candidate.availability_days} days."
        )

        summary = (
            f"{candidate.full_name} showed {interest_level} interest, with a {sentiment} tone. "
            f"Primary overlap came from {', '.join(matched_skills[:3]) if matched_skills else 'limited skill overlap'}."
        )

        return ConversationDraft(
            consent_response=(
                "Yes, I can spare a few minutes to learn more about the opportunity."
                if consent_given
                else "I appreciate the message, but I am not the right fit to continue right now."
            ),
            interest_response=interest_response,
            salary_response=salary_response,
            availability_response=availability_response,
            summary=summary,
            assessment=ConversationAssessment(
                consent_given=consent_given,
                interest_level=interest_level,
                sentiment=sentiment,
                confidence=confidence,
                specificity=specificity,
            ),
        )


class ConversationService:
    def __init__(
        self,
        *,
        llm: BaseConversationLLM | None = None,
        storage_dir: str | None = None,
    ) -> None:
        settings = get_settings()
        self.storage_dir = Path(storage_dir or settings.conversation_log_path)
        self.llm = llm or self._build_default_llm()

    def _build_default_llm(self) -> BaseConversationLLM:
        settings = get_settings()
        if settings.groq_api_key:
            return GroqConversationLLM()
        return MockConversationLLM()

    def _build_transcript(
        self,
        recruiter_prompts: dict[str, str],
        draft: ConversationDraft,
    ) -> list[ConversationTurn]:
        return [
            ConversationTurn(stage="consent", speaker="recruiter", message=recruiter_prompts["consent"]),
            ConversationTurn(stage="consent", speaker="candidate", message=draft.consent_response),
            ConversationTurn(stage="interest", speaker="recruiter", message=recruiter_prompts["interest"]),
            ConversationTurn(stage="interest", speaker="candidate", message=draft.interest_response),
            ConversationTurn(stage="salary", speaker="recruiter", message=recruiter_prompts["salary"]),
            ConversationTurn(stage="salary", speaker="candidate", message=draft.salary_response),
            ConversationTurn(
                stage="availability",
                speaker="recruiter",
                message=recruiter_prompts["availability"],
            ),
            ConversationTurn(
                stage="availability",
                speaker="candidate",
                message=draft.availability_response,
            ),
        ]

    def _save_conversation(self, conversation: CandidateConversation) -> str:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.storage_dir / f"{conversation.conversation_id}.json"
        output_path.write_text(
            conversation.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return str(output_path)

    def simulate_conversation(
        self,
        candidate: Candidate,
        parsed_jd: ParsedJobDescription,
        *,
        recruiter_name: str = "Talent Scout Bot",
    ) -> CandidateConversation:
        recruiter_prompts = build_recruiter_prompts(parsed_jd, recruiter_name)
        try:
            draft = self.llm.generate_draft(candidate, parsed_jd, recruiter_prompts)
        except Exception:
            # Fall back to local deterministic simulation so ranking does not fail
            # when an external provider is unavailable or slow.
            fallback_llm = MockConversationLLM()
            draft = fallback_llm.generate_draft(candidate, parsed_jd, recruiter_prompts)

        conversation = CandidateConversation(
            conversation_id=f"{candidate.id}-{uuid4().hex[:8]}",
            candidate_id=candidate.id,
            full_name=candidate.full_name,
            role_title=candidate.role_title,
            provider=self.llm.provider,
            model=self.llm.model_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            summary=draft.summary,
            transcript=self._build_transcript(recruiter_prompts, draft),
            signals=ConversationSignals(
                consent_given=draft.assessment.consent_given,
                interest_level=draft.assessment.interest_level,
                sentiment=draft.assessment.sentiment,
                confidence=draft.assessment.confidence,
                specificity=draft.assessment.specificity,
                salary_expectation_usd=candidate.expected_salary_usd,
                salary_alignment=calculate_salary_alignment(
                    candidate.expected_salary_usd,
                    parsed_jd.salary_range_usd,
                ),
                availability_days=candidate.availability_days,
            ),
            storage_path="",
        )

        storage_path = self._save_conversation(conversation)
        conversation.storage_path = storage_path
        Path(storage_path).write_text(
            conversation.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return conversation
