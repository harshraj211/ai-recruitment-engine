import json
from pathlib import Path

from app.schemas.conversation import ConversationAssessment, ConversationDraft
from app.services.candidate_store import load_candidate_lookup
from app.services.conversation_service import (
    ConversationService,
    MockConversationLLM,
    calculate_salary_alignment,
)
from app.services.jd_parser import parse_job_description


class StubConversationLLM:
    provider = "stub"
    model_name = "stub-model"

    def generate_draft(self, candidate, parsed_jd, recruiter_prompts):
        return ConversationDraft(
            consent_response="Yes, I can chat for a few minutes.",
            interest_response="The role matches my Python and FastAPI experience quite well.",
            salary_response="The shared budget is in line with what I am looking for.",
            availability_response="I could start in about 21 days.",
            summary="The candidate sounds interested and practical about next steps.",
            assessment=ConversationAssessment(
                consent_given=True,
                interest_level="high",
                sentiment="positive",
                confidence="high",
                specificity="high",
            ),
        )


def test_calculate_salary_alignment_handles_common_cases() -> None:
    assert calculate_salary_alignment(55000, [50000, 65000]) == "aligned"
    assert calculate_salary_alignment(45000, [50000, 65000]) == "below_range"
    assert calculate_salary_alignment(70000, [50000, 65000]) == "above_range"
    assert calculate_salary_alignment(55000, []) == "unknown"


def test_conversation_service_saves_transcript_and_signals(tmp_path: Path) -> None:
    parsed_jd = parse_job_description(
        """
        We are hiring a Machine Learning Engineer with 4+ years of experience.
        Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search.
        Budget: $50,000 - $65,000 annually.
        """
    )
    candidate = load_candidate_lookup()["cand-002"]
    service = ConversationService(
        llm=StubConversationLLM(),
        storage_dir=str(tmp_path),
    )

    conversation = service.simulate_conversation(candidate, parsed_jd)

    assert conversation.provider == "stub"
    assert conversation.signals.salary_alignment == "aligned"
    assert len(conversation.transcript) == 8
    assert Path(conversation.storage_path).exists()

    stored_payload = json.loads(Path(conversation.storage_path).read_text(encoding="utf-8"))
    assert stored_payload["candidate_id"] == "cand-002"
    assert stored_payload["signals"]["interest_level"] == "high"


def test_mock_conversation_llm_produces_candidate_responses() -> None:
    parsed_jd = parse_job_description(
        """
        We are hiring a Senior Machine Learning Engineer for an AI platform.
        You should have 4+ years of experience.
        Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search.
        Budget: $50,000 - $65,000 annually.
        """
    )
    candidate = load_candidate_lookup()["cand-002"]
    draft = MockConversationLLM().generate_draft(
        candidate,
        parsed_jd,
        recruiter_prompts={
            "consent": "consent prompt",
            "interest": "interest prompt",
            "salary": "salary prompt",
            "availability": "availability prompt",
        },
    )

    assert draft.assessment.interest_level == "high"
    assert "$54,000" in draft.salary_response
    assert "Python" in draft.interest_response
