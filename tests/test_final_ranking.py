from pathlib import Path

from app.schemas.conversation import CandidateConversation, ConversationSignals
from app.schemas.final_ranking import FinalCandidateRanking
from app.services.final_ranking import FinalRankingService
from app.services.jd_parser import parse_job_description


class FakeVectorStore:
    def search_parsed_job(self, parsed_jd, top_k=5):
        from app.schemas.semantic_search import SemanticSearchResult

        return [
            SemanticSearchResult(
                candidate_id="cand-002",
                full_name="Rohan Mehta",
                role_title="Machine Learning Engineer",
                total_experience_years=5.0,
                skills=["Python", "FastAPI", "PyTorch", "Docker", "MLflow"],
                similarity_score=0.74,
                profile_summary="summary",
            ),
            SemanticSearchResult(
                candidate_id="cand-007",
                full_name="Meera Iyer",
                role_title="MLOps Engineer",
                total_experience_years=7.0,
                skills=["Python", "Docker", "AWS", "MLflow"],
                similarity_score=0.66,
                profile_summary="summary",
            ),
            SemanticSearchResult(
                candidate_id="cand-015",
                full_name="Sneha Patel",
                role_title="Computer Vision Engineer",
                total_experience_years=5.0,
                skills=["Python", "PyTorch", "FastAPI", "Docker"],
                similarity_score=0.65,
                profile_summary="summary",
            ),
        ]


class FakeConversationService:
    def simulate_conversation(self, candidate, parsed_jd, recruiter_name="Talent Scout Bot"):
        if candidate.id == "cand-002":
            return CandidateConversation(
                conversation_id="conv-002",
                candidate_id="cand-002",
                full_name=candidate.full_name,
                role_title=candidate.role_title,
                provider="mock",
                model="mock-local",
                created_at="2026-04-25T00:00:00+00:00",
                summary="Strong interest",
                transcript=[],
                signals=ConversationSignals(
                    consent_given=True,
                    interest_level="high",
                    sentiment="positive",
                    confidence="high",
                    specificity="high",
                    salary_expectation_usd=54000,
                    salary_alignment="aligned",
                    availability_days=21,
                ),
                storage_path=str(Path("data/conversations/conv-002.json")),
            )

        if candidate.id == "cand-015":
            return CandidateConversation(
                conversation_id="conv-015",
                candidate_id="cand-015",
                full_name=candidate.full_name,
                role_title=candidate.role_title,
                provider="mock",
                model="mock-local",
                created_at="2026-04-25T00:00:00+00:00",
                summary="Strong interest",
                transcript=[],
                signals=ConversationSignals(
                    consent_given=True,
                    interest_level="high",
                    sentiment="positive",
                    confidence="high",
                    specificity="high",
                    salary_expectation_usd=55000,
                    salary_alignment="aligned",
                    availability_days=21,
                ),
                storage_path=str(Path("data/conversations/conv-015.json")),
            )

        return CandidateConversation(
            conversation_id="conv-007",
            candidate_id="cand-007",
            full_name=candidate.full_name,
            role_title=candidate.role_title,
            provider="mock",
            model="mock-local",
            created_at="2026-04-25T00:00:00+00:00",
            summary="Moderate interest",
            transcript=[],
            signals=ConversationSignals(
                consent_given=True,
                interest_level="medium",
                sentiment="neutral",
                confidence="medium",
                specificity="medium",
                salary_expectation_usd=76000,
                salary_alignment="above_range",
                availability_days=60,
            ),
            storage_path=str(Path("data/conversations/conv-007.json")),
        )


def test_final_ranking_combines_match_and_interest_scores() -> None:
    parsed_jd = parse_job_description(
        """
        We are hiring a Senior Machine Learning Engineer for our talent intelligence platform.
        You should have 4+ years of experience building production APIs and ML services.
        Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search.
        Budget: $50,000 - $65,000 annually.
        This is a remote role.
        """
    )

    rankings = FinalRankingService(
        vector_store=FakeVectorStore(),
        conversation_service=FakeConversationService(),
    ).rank_candidates(parsed_jd, top_k_search=3, top_k_final=2)

    assert len(rankings) == 2
    assert rankings[0].candidate_id == "cand-002"
    assert rankings[0].final_score == 89.5
    assert rankings[1].candidate_id == "cand-015"
    assert rankings[1].final_score < rankings[0].final_score
    assert rankings[0].rank == 1
    assert "Final Score 89.5%" in rankings[0].final_explanation


def test_final_ranking_results_have_nested_scores() -> None:
    parsed_jd = parse_job_description(
        """
        Need a Machine Learning Engineer with Python, FastAPI, PyTorch, Docker, AWS, MLflow,
        and vector search.
        """
    )

    rankings = FinalRankingService(
        vector_store=FakeVectorStore(),
        conversation_service=FakeConversationService(),
    ).rank_candidates(parsed_jd, top_k_search=3, top_k_final=1)

    result = rankings[0]
    assert isinstance(result, FinalCandidateRanking)
    assert result.match_result.match_score > 0
    assert result.interest_result.interest_score > 0


def test_final_ranking_run_returns_pipeline_counts() -> None:
    parsed_jd = parse_job_description(
        """
        Need a Machine Learning Engineer with Python, FastAPI, PyTorch, Docker, AWS, MLflow,
        and vector search.
        """
    )

    run = FinalRankingService(
        vector_store=FakeVectorStore(),
        conversation_service=FakeConversationService(),
    ).run_ranking(parsed_jd, top_k_search=3, top_k_final=2)

    assert run.total_candidates_retrieved == 3
    assert run.total_candidates_ranked == 3
    assert len(run.rankings) == 2
