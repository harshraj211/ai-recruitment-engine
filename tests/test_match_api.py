from fastapi.testclient import TestClient

from app.api.routes.matching import get_match_pipeline_service
from app.main import app
from app.schemas.final_ranking import FinalCandidateRanking
from app.schemas.interest_scoring import CandidateInterestResult, InterestScoreBreakdown
from app.schemas.job_description import ParsedJobDescription
from app.schemas.match_scoring import CandidateMatchResult
from app.schemas.pipeline import MatchPipelineResult


class StubPipelineService:
    async def run_async(self, job_description, top_k_search=10, top_k_final=5, page=1, page_size=5, include_outreach=False):
        return MatchPipelineResult(
            parsed_job_description=ParsedJobDescription(
                raw_text=job_description,
                role_title="Machine Learning Engineer",
                seniority="senior",
                min_experience_years=4.0,
                skills=["Python", "FastAPI", "PyTorch"],
                mandatory_skills=["Python", "FastAPI"],
                nice_to_have_skills=["PyTorch"],
                salary_range_usd=[],
                work_mode=None,
            ),
            rankings=[
                FinalCandidateRanking(
                    candidate_id="cand-002",
                    full_name="Rohan Mehta",
                    role_title="Machine Learning Engineer",
                    candidate_name="Rohan Mehta",
                    match_score=88.0,
                    interest_score=74.0,
                    bm25_score=0.91,
                    cross_encoder_score=82.0,
                    flight_risk_score=68.0,
                    final_score=83.0,
                    rank=1,
                    match_result=CandidateMatchResult(
                        candidate_id="cand-002",
                        full_name="Rohan Mehta",
                        role_title="Machine Learning Engineer",
                        total_experience_years=5.0,
                        match_score=88.0,
                        skill_match_score=0.86,
                        core_skill_score=1.0,
                        secondary_skill_score=0.72,
                        experience_match_score=1.0,
                        role_alignment_score=1.0,
                        trajectory_boost_score=0.0,
                        matched_skills=["Python", "FastAPI", "PyTorch"],
                        missing_skills=["Vector Search"],
                        matched_core_skills=["Python", "FastAPI"],
                        missing_core_skills=["Vector Search"],
                        matched_secondary_skills=["PyTorch"],
                        missing_secondary_skills=[],
                        skill_alignment_details=["Python: exact (1.00)"],
                        explanation="Match explanation",
                        semantic_similarity_score=0.75,
                        cross_encoder_score=0.82,
                    ),
                    interest_result=CandidateInterestResult(
                        candidate_id="cand-002",
                        full_name="Rohan Mehta",
                        role_title="Machine Learning Engineer",
                        interest_score=74.0,
                        flight_risk_score=68.0,
                        breakdown=InterestScoreBreakdown(
                            tenure_score=0.8,
                            salary_alignment_score=0.85,
                            availability_score=0.85,
                            stagnation_score=0.4,
                            promotion_likelihood_score=0.2,
                            role_relevance_score=1.0,
                            engagement_probability_score=1.0,
                        ),
                        salary_alignment="aligned",
                        availability_days=21,
                        explanation="Interest explanation",
                        provider="deterministic",
                    ),
                    summary="Strong backend ML engineer with solid switch likelihood.",
                    missing_skills=["Vector Search"],
                    recommendation="Advance to recruiter screen.",
                    final_explanation="Final explanation",
                )
            ][:top_k_final],
            total_candidates_retrieved=5,
            total_candidates_ranked=5,
            total_candidates_returned=min(top_k_final, 1),
            page=page,
            page_size=page_size,
            total_pages=1,
        )


def test_match_endpoint_returns_ranked_candidates() -> None:
    app.dependency_overrides[get_match_pipeline_service] = lambda: StubPipelineService()
    client = TestClient(app)

    response = client.post(
        "/api/v1/match",
        json={
            "job_description": (
                "We are hiring a Senior Machine Learning Engineer for our talent intelligence "
                "platform. You should have 4+ years of experience building production APIs and "
                "ML services. Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, "
                "and vector search."
            ),
            "top_k_search": 5,
            "top_k_final": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["parsed_job_description"]["role_title"] == "Machine Learning Engineer"
    assert payload["rankings"][0]["candidate_id"] == "cand-002"
    assert payload["rankings"][0]["bm25_score"] == 0.91
    assert payload["rankings"][0]["cross_encoder_score"] == 82.0
    assert payload["rankings"][0]["flight_risk_score"] == 68.0
    assert payload["total_candidates_considered"] == 5
    assert payload["page"] == 1
    assert payload["total_pages"] == 1

    app.dependency_overrides.clear()


def test_match_endpoint_validates_request_body() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/match",
        json={"job_description": "too short"},
    )
    assert response.status_code == 422
