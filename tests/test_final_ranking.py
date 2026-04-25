from app.schemas.interest_scoring import CandidateInterestResult, InterestScoreBreakdown
from app.schemas.outreach import RecruiterOutreach
from app.schemas.semantic_search import SemanticSearchResult
from app.services.final_ranking import FinalRankingService
from app.services.jd_parser import parse_job_description


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
                skills=["Python", "FastAPI", "PyTorch", "Docker", "MLflow"],
                similarity_score=0.91,
                semantic_similarity_score=0.88,
                keyword_match_score=0.84,
                bm25_score=0.84,
                dense_profile_score=0.86,
                dense_skill_score=0.89,
                rrf_score=0.91,
                profile_summary="summary",
            ),
            SemanticSearchResult(
                candidate_id="cand-015",
                full_name="Sneha Patel",
                role_title="Computer Vision Engineer",
                total_experience_years=5.0,
                skills=["Python", "PyTorch", "FastAPI", "Docker"],
                similarity_score=0.73,
                semantic_similarity_score=0.72,
                keyword_match_score=0.66,
                bm25_score=0.66,
                dense_profile_score=0.71,
                dense_skill_score=0.69,
                rrf_score=0.73,
                profile_summary="summary",
            ),
        ]


class FakeCrossEncoderService:
    async def score_pairs_async(self, pairs):
        return [0.92, 0.61][: len(pairs)]


class FakeEngagementService:
    async def score_candidate_async(self, candidate, parsed_jd):
        interest_lookup = {
            "cand-002": 78.0,
            "cand-015": 70.0,
        }
        flight_risk_lookup = {
            "cand-002": 66.0,
            "cand-015": 61.0,
        }
        return CandidateInterestResult(
            candidate_id=candidate.id,
            full_name=candidate.full_name,
            role_title=candidate.role_title,
            interest_score=interest_lookup[candidate.id],
            flight_risk_score=flight_risk_lookup[candidate.id],
            breakdown=InterestScoreBreakdown(
                tenure_score=0.8,
                salary_alignment_score=0.85,
                availability_score=0.85,
                stagnation_score=0.35,
                promotion_likelihood_score=0.25,
                role_relevance_score=0.9,
                engagement_probability_score=0.8,
            ),
            salary_alignment="aligned",
            availability_days=21,
            explanation="Interest explanation",
            provider="deterministic",
        )


class FakeCommunicationService:
    async def generate_summary(self, candidate, parsed_jd, match_result, interest_result):
        return (
            f"{candidate.full_name} looks strong on core backend ML fit with manageable risk.",
            "deterministic",
            None,
        )

    async def generate_outreach(self, candidate, parsed_jd, match_result, interest_result):
        return RecruiterOutreach(
            message=f"Hi, I would love to speak about our {parsed_jd.role_title} opening.",
            provider="deterministic",
            model="rule-based",
        )


def test_final_ranking_combines_match_interest_and_cross_encoder() -> None:
    parsed_jd = parse_job_description(
        """
        We are hiring a Senior Machine Learning Engineer for our talent intelligence platform.
        Must have Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search.
        Budget: $50,000 - $65,000 annually.
        This is a remote role.
        """
    )

    rankings = FinalRankingService(
        vector_store=FakeVectorStore(),
        engagement_service=FakeEngagementService(),
        communication_service=FakeCommunicationService(),
        cross_encoder_service=FakeCrossEncoderService(),
    ).rank_candidates(parsed_jd, top_k_search=2, top_k_final=2, include_outreach=True)

    assert len(rankings) == 2
    assert rankings[0].candidate_id == "cand-002"
    assert rankings[0].final_score > rankings[1].final_score
    assert rankings[0].cross_encoder_score >= rankings[1].cross_encoder_score
    assert rankings[0].flight_risk_score == 66.0
    assert rankings[0].recruiter_outreach is not None


def test_final_ranking_results_include_flat_and_nested_outputs() -> None:
    parsed_jd = parse_job_description(
        """
        Need a Machine Learning Engineer with Python, FastAPI, PyTorch, Docker, AWS, MLflow,
        and vector search.
        """
    )

    rankings = FinalRankingService(
        vector_store=FakeVectorStore(),
        engagement_service=FakeEngagementService(),
        communication_service=FakeCommunicationService(),
        cross_encoder_service=FakeCrossEncoderService(),
    ).rank_candidates(parsed_jd, top_k_search=2, top_k_final=1)

    result = rankings[0]
    assert result.match_result.match_score > 0
    assert result.interest_result.interest_score > 0
    assert result.summary
    assert result.recommendation


def test_final_ranking_run_returns_pipeline_counts() -> None:
    parsed_jd = parse_job_description(
        """
        Need a Machine Learning Engineer with Python, FastAPI, PyTorch, Docker, AWS, MLflow,
        and vector search.
        """
    )

    run = FinalRankingService(
        vector_store=FakeVectorStore(),
        engagement_service=FakeEngagementService(),
        communication_service=FakeCommunicationService(),
        cross_encoder_service=FakeCrossEncoderService(),
    ).run_ranking(parsed_jd, top_k_search=2, top_k_final=2, page=1, page_size=2)

    assert run.total_candidates_retrieved == 2
    assert run.total_candidates_ranked == 2
    assert len(run.rankings) == 2
    assert run.total_pages == 1
