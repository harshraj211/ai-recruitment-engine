from app.schemas.conversation import CandidateConversation, ConversationSignals, ConversationTurn
from app.services.interest_scoring import (
    map_availability_score,
    map_salary_match_score,
    score_candidate_interest,
)


def build_conversation(
    *,
    consent_given: bool = True,
    sentiment: str = "positive",
    confidence: str = "high",
    specificity: str = "high",
    salary_alignment: str = "aligned",
    availability_days: int | None = 21,
) -> CandidateConversation:
    return CandidateConversation(
        conversation_id="conv-001",
        candidate_id="cand-001",
        full_name="Aanya Sharma",
        role_title="Senior Data Scientist",
        provider="mock",
        model="mock-local",
        created_at="2026-04-25T00:00:00+00:00",
        summary="Sample summary",
        transcript=[
            ConversationTurn(stage="consent", speaker="recruiter", message="hi"),
            ConversationTurn(stage="consent", speaker="candidate", message="yes"),
        ],
        signals=ConversationSignals(
            consent_given=consent_given,
            interest_level="high",
            sentiment=sentiment,
            confidence=confidence,
            specificity=specificity,
            salary_expectation_usd=62000,
            salary_alignment=salary_alignment,
            availability_days=availability_days,
        ),
        storage_path="data/conversations/conv-001.json",
    )


def test_map_salary_match_score_treats_below_range_as_positive() -> None:
    assert map_salary_match_score("aligned") == 1.0
    assert map_salary_match_score("below_range") == 1.0
    assert map_salary_match_score("above_range") == 0.0
    assert map_salary_match_score("unknown") == 0.5


def test_map_availability_score_uses_simple_bands() -> None:
    assert map_availability_score(21) == 1.0
    assert map_availability_score(45) == 0.5
    assert map_availability_score(75) == 0.0
    assert map_availability_score(None) == 0.5


def test_score_candidate_interest_returns_full_score_for_strong_signals() -> None:
    conversation = build_conversation()

    result = score_candidate_interest(conversation)

    assert result.interest_score == 100.0
    assert result.breakdown.sentiment_score == 1.0
    assert "Final Interest Score 100.0%" in result.explanation


def test_score_candidate_interest_zeroes_out_when_candidate_declines() -> None:
    conversation = build_conversation(consent_given=False)

    result = score_candidate_interest(conversation)

    assert result.interest_score == 0.0
    assert result.breakdown.salary_match_score == 0.0
    assert "did not consent" in result.explanation


def test_score_candidate_interest_matches_weighted_formula_for_mixed_signals() -> None:
    conversation = build_conversation(
        sentiment="neutral",
        confidence="medium",
        specificity="medium",
        salary_alignment="unknown",
        availability_days=45,
    )

    result = score_candidate_interest(conversation)

    assert result.interest_score == 50.0
    assert result.breakdown.availability_score == 0.5
