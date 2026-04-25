import asyncio

from app.services.candidate_store import load_candidate_lookup
from app.services.conversation_service import (
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
    parsed_jd = parse_job_description(
        """
        We are hiring a Senior Machine Learning Engineer.
        Required skills: Python, FastAPI, PyTorch, Docker, AWS, and vector search.
        Budget: $50,000 - $65,000 annually.
        """
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


def test_deterministic_communication_llm_generates_recruiter_ready_text() -> None:
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

    service = RecruiterCommunicationService(llm=DeterministicCommunicationLLM())

    outreach = asyncio.run(
        service.generate_outreach(candidate, parsed_jd, match_result, interest_result)
    )
    summary, provider, fallback_reason = asyncio.run(
        service.generate_summary(candidate, parsed_jd, match_result, interest_result)
    )

    assert outreach.provider == "deterministic"
    assert "Machine Learning Engineer" in outreach.message
    assert provider == "deterministic"
    assert fallback_reason is None
    assert summary
