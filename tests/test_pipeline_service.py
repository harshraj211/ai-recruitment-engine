from app.schemas.conversation import CandidateConversation, ConversationSignals
from app.schemas.semantic_search import SemanticSearchResult
from app.services.final_ranking import FinalRankingService
from app.services.pipeline_service import MatchPipelineService
from pathlib import Path


class FakeVectorStore:
    def search_parsed_job(self, parsed_jd, top_k=5):
        return [
            SemanticSearchResult(
                candidate_id="cand-002",
                full_name="Rohan Mehta",
                role_title="Machine Learning Engineer",
                total_experience_years=5.0,
                skills=["Python", "FastAPI", "PyTorch"],
                similarity_score=0.74,
                profile_summary="summary",
            )
        ]


class FakeConversationService:
    def simulate_conversation(self, candidate, parsed_jd, recruiter_name="Talent Scout Bot"):
        return CandidateConversation(
            conversation_id="conv-002",
            candidate_id=candidate.id,
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


def test_pipeline_service_runs_end_to_end_and_returns_counts() -> None:
    service = MatchPipelineService(
        final_ranking_service=FinalRankingService(
            vector_store=FakeVectorStore(),
            conversation_service=FakeConversationService(),
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
