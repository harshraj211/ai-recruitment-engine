import asyncio

from app.schemas.final_ranking import FinalCandidateRanking
from app.services.candidate_store import load_candidate_lookup
from app.services.interest_scoring import PredictiveEngagementService
from app.services.jd_parser import parse_job_description
from app.services.match_scoring import score_candidate_match
from app.services.response_validation import ResponseValidationService


def test_response_validation_reconciles_flat_fields_and_summary() -> None:
    candidate = load_candidate_lookup()["cand-002"]
    parsed_jd = parse_job_description(
        """
        We are hiring a Senior Machine Learning Engineer.
        Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search.
        Nice to have Kubernetes.
        Budget: $50,000 - $65,000 annually.
        """
    )
    match_result = score_candidate_match(parsed_jd, candidate, cross_encoder_score=0.82)
    interest_result = PredictiveEngagementService().score_candidate(candidate, parsed_jd)
    ranking = FinalCandidateRanking(
        candidate_id=match_result.candidate_id,
        full_name=match_result.full_name,
        role_title=match_result.role_title,
        candidate_name="Wrong Name",
        match_score=match_result.match_score,
        interest_score=100.0,
        bm25_score=0.91,
        cross_encoder_score=0.82,
        flight_risk_score=0.0,
        final_score=99.0,
        rank=1,
        match_result=match_result,
        interest_result=interest_result,
        summary="Strong fit overall, but missing AWS.",
        missing_skills=["Vector Search"],
        recommendation="Advance to recruiter screen.",
        final_explanation="Outdated explanation",
    )

    corrected = asyncio.run(
        ResponseValidationService().validate_candidate_ranking(
            ranking,
            candidate=candidate,
            parsed_jd=parsed_jd,
        )
    )

    assert corrected.candidate_name == corrected.full_name
    assert corrected.cross_encoder_score == 82.0
    assert corrected.interest_score == corrected.interest_result.interest_score
    assert corrected.flight_risk_score == corrected.interest_result.flight_risk_score
    assert corrected.missing_skills == corrected.match_result.missing_skills
    assert "AWS" not in corrected.summary
    assert "Kubernetes" in corrected.missing_skills
