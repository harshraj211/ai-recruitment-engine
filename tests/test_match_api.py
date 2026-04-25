from fastapi.testclient import TestClient

from app.api.routes.matching import get_match_pipeline_service
from app.main import app
from app.schemas.final_ranking import FinalCandidateRanking
from app.schemas.interest_scoring import CandidateInterestResult, InterestScoreBreakdown
from app.schemas.match_scoring import CandidateMatchResult
from app.schemas.pipeline import MatchPipelineResult
from app.schemas.job_description import ParsedJobDescription


class StubPipelineService:
    def run(self, job_description, top_k_search=5, top_k_final=5):
        return MatchPipelineResult(
            parsed_job_description=ParsedJobDescription(
                raw_text=job_description,
                role_title="Machine Learning Engineer",
                seniority="senior",
                min_experience_years=4.0,
                skills=["Python", "FastAPI", "PyTorch"],
                salary_range_usd=[],
                work_mode=None,
            ),
            rankings=[
                FinalCandidateRanking(
                    candidate_id="cand-002",
                    full_name="Rohan Mehta",
                    role_title="Machine Learning Engineer",
                    final_score=89.5,
                    rank=1,
                    match_result=CandidateMatchResult(
                        candidate_id="cand-002",
                        full_name="Rohan Mehta",
                        role_title="Machine Learning Engineer",
                        total_experience_years=5.0,
                        match_score=82.5,
                        skill_match_score=0.75,
                        experience_match_score=1.0,
                        matched_skills=["Python", "FastAPI", "PyTorch", "Docker", "AWS", "MLflow"],
                        missing_skills=["Machine Learning", "Vector Search"],
                        explanation="Match explanation",
                        semantic_similarity_score=0.7471,
                    ),
                    interest_result=CandidateInterestResult(
                        candidate_id="cand-002",
                        full_name="Rohan Mehta",
                        role_title="Machine Learning Engineer",
                        interest_score=100.0,
                        breakdown=InterestScoreBreakdown(
                            sentiment_score=1.0,
                            confidence_score=1.0,
                            specificity_score=1.0,
                            salary_match_score=1.0,
                            availability_score=1.0,
                        ),
                        explanation="Interest explanation",
                        conversation_id="conv-002",
                        provider="mock",
                    ),
                    final_explanation="Final explanation",
                )
            ][:top_k_final],
            total_candidates_retrieved=5,
            total_candidates_ranked=5,
            total_candidates_returned=min(top_k_final, 1),
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
    assert payload["rankings"][0]["final_score"] == 89.5
    assert payload["total_candidates_considered"] == 5
    assert payload["total_candidates_retrieved"] == 5
    assert payload["total_candidates_returned"] == 1

    app.dependency_overrides.clear()


def test_match_endpoint_validates_request_body() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/match",
        json={"job_description": "too short"},
    )
    assert response.status_code == 422
