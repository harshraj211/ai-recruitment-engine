import asyncio

from app.schemas.job_description import ParsedJobDescription
from app.services.candidate_store import load_candidate_lookup
from app.services.conversation_service import (
    BaseCommunicationLLM,
    DeterministicCommunicationLLM,
    RecruiterCommunicationService,
    build_recruiter_outreach_prompt,
    build_summary_prompt,
    calculate_salary_alignment,
)
from app.services.interest_scoring import PredictiveEngagementService
from app.services.jd_parser import parse_job_description
from app.services.match_scoring import score_candidate_match


def test_calculate_salary_alignment_handles_common_cases() -> None:
    assert calculate_salary_alignment(55000, [50000, 65000]) == "aligned"
    assert calculate_salary_alignment(45000, [50000, 65000]) == "below_range"
    assert calculate_salary_alignment(70000, [50000, 65000]) == "above_range"
    assert calculate_salary_alignment(55000, []) == "unknown"


def test_build_prompts_mask_candidate_name() -> None:
    candidate = load_candidate_lookup()["cand-002"]
    parsed_jd = ParsedJobDescription(
        raw_text="We are hiring an Applied AI Engineer.",
        role_title="Applied AI Engineer",
        mandatory_skills=["Python", "FastAPI", "PyTorch", "Docker", "AWS", "Vector Search"],
        nice_to_have_skills=["MLflow"],
        salary_range_usd=[50000, 65000],
    )
    match_result = score_candidate_match(parsed_jd, candidate)
    interest_result = PredictiveEngagementService().score_candidate(candidate, parsed_jd)

    outreach_prompt = build_recruiter_outreach_prompt(
        candidate,
        parsed_jd,
        match_result,
        interest_result,
    )
    summary_prompt = build_summary_prompt(
        candidate,
        parsed_jd,
        match_result,
        interest_result,
    )

    assert outreach_prompt["candidate"]["candidate_label"] != candidate.full_name
    assert summary_prompt["candidate"]["candidate_label"] != candidate.full_name
    assert outreach_prompt["candidate"]["current_role_title"] == candidate.role_title
    assert "role_title" not in outreach_prompt["candidate"]
    assert outreach_prompt["role"]["target_role_title"] == "Applied AI Engineer"
    assert summary_prompt["signals"]["matched_skills"]
    assert summary_prompt["signals"]["missing_skills"] == ["Vector Search"]
    assert any("Do NOT invent missing skills" in rule for rule in summary_prompt["rules"])


def test_deterministic_communication_llm_generates_recruiter_ready_text() -> None:
    candidate = load_candidate_lookup()["cand-002"]
    parsed_jd = ParsedJobDescription(
        raw_text="We are hiring an Applied AI Engineer.",
        role_title="Applied AI Engineer",
        mandatory_skills=["Python", "FastAPI", "PyTorch", "Docker", "AWS", "Vector Search"],
        nice_to_have_skills=["MLflow"],
        salary_range_usd=[50000, 65000],
    )
    match_result = score_candidate_match(parsed_jd, candidate)
    interest_result = PredictiveEngagementService().score_candidate(candidate, parsed_jd)

    service = RecruiterCommunicationService(llm=DeterministicCommunicationLLM())

    outreach = asyncio.run(
        service.generate_outreach(candidate, parsed_jd, match_result, interest_result)
    )
    summary, provider, fallback_reason = asyncio.run(
        service.generate_summary(candidate, parsed_jd, match_result, interest_result)
    )

    assert outreach.provider == "deterministic"
    assert "Applied AI Engineer" in outreach.message
    assert candidate.role_title not in outreach.message
    assert provider == "deterministic"
    assert fallback_reason is None
    assert summary


class HallucinatingSummaryLLM(BaseCommunicationLLM):
    provider = "groq"
    model_name = "fake-groq"

    async def generate_text(self, prompt: dict, *, max_tokens: int) -> str:
        return "Strong fit overall, but missing AWS."


def test_summary_falls_back_when_llm_invents_missing_skill() -> None:
    candidate = load_candidate_lookup()["cand-002"]
    parsed_jd = parse_job_description(
        """
        We are hiring a Senior Machine Learning Engineer.
        Required skills: Python, FastAPI, PyTorch, Docker, AWS, and vector search.
        Budget: $50,000 - $65,000 annually.
        """
    )
    match_result = score_candidate_match(parsed_jd, candidate)
    interest_result = PredictiveEngagementService().score_candidate(candidate, parsed_jd)

    service = RecruiterCommunicationService(llm=HallucinatingSummaryLLM())

    summary, provider, fallback_reason = asyncio.run(
        service.generate_summary(candidate, parsed_jd, match_result, interest_result)
    )

    assert provider == "deterministic"
    assert fallback_reason is not None
    assert "missing AWS" not in summary
