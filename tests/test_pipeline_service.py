from app.schemas.interest_scoring import CandidateInterestResult, InterestScoreBreakdown
from app.schemas.semantic_search import SemanticSearchResult
from app.services.final_ranking import FinalRankingService
from app.services.pipeline_service import MatchPipelineService


class FakeVectorStore:
    def search_parsed_job(self, parsed_jd, top_k=5):
        return self._results()[:top_k]

    async def search_parsed_job_async(self, parsed_jd, top_k=5):
        return self._results()[:top_k]

    @staticmethod
    def _results():
        return [
            SemanticSearchResult(
                candidate_id="cand-002",
                full_name="Rohan Mehta",
                role_title="Machine Learning Engineer",
                total_experience_years=5.0,
                skills=["Python", "FastAPI", "PyTorch"],
                similarity_score=0.74,
                semantic_similarity_score=0.71,
                keyword_match_score=0.68,
                bm25_score=0.68,
                dense_profile_score=0.73,
                dense_skill_score=0.72,
                rrf_score=0.74,
                profile_summary="summary",
            )
        ]


class FakeCrossEncoderService:
    async def score_pairs_async(self, pairs):
        return [0.9 for _ in pairs]


class FakeEngagementService:
    async def score_candidate_async(self, candidate, parsed_jd):
        return CandidateInterestResult(
            candidate_id=candidate.id,
            full_name=candidate.full_name,
            role_title=candidate.role_title,
            interest_score=76.0,
            flight_risk_score=65.0,
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
        )


class FakeCommunicationService:
    async def generate_summary(self, candidate, parsed_jd, match_result, interest_result):
        return ("Strong fit summary.", "deterministic", None)

    async def generate_outreach(self, candidate, parsed_jd, match_result, interest_result):
        return None


def test_pipeline_service_runs_end_to_end_and_returns_counts() -> None:
    service = MatchPipelineService(
        final_ranking_service=FinalRankingService(
            vector_store=FakeVectorStore(),
            engagement_service=FakeEngagementService(),
            communication_service=FakeCommunicationService(),
            cross_encoder_service=FakeCrossEncoderService(),
        )
    )

    result = service.run(
        "We are hiring a Senior Machine Learning Engineer for our platform. Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search.",
        top_k_search=3,
        top_k_final=1,
    )

    assert result.parsed_job_description.role_title == "Machine Learning Engineer"
    assert result.total_candidates_retrieved == 1
    assert result.total_candidates_ranked == 1
    assert result.total_candidates_returned == 1
    assert result.rankings[0].candidate_id == "cand-002"
