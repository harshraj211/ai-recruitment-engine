from app.schemas.conversation import CandidateConversation
from app.schemas.interest_scoring import CandidateInterestResult, InterestScoreBreakdown

SENTIMENT_WEIGHTS = {
    "positive": 1.0,
    "neutral": 0.5,
    "negative": 0.0,
}

CONFIDENCE_WEIGHTS = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.0,
}

SPECIFICITY_WEIGHTS = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.0,
}

SALARY_ALIGNMENT_WEIGHTS = {
    "aligned": 1.0,
    "below_range": 1.0,
    "above_range": 0.0,
    "unknown": 0.5,
}

SENTIMENT_FACTOR = 3
CONFIDENCE_FACTOR = 2
SPECIFICITY_FACTOR = 2
SALARY_FACTOR = 2
AVAILABILITY_FACTOR = 1
TOTAL_FACTOR = (
    SENTIMENT_FACTOR
    + CONFIDENCE_FACTOR
    + SPECIFICITY_FACTOR
    + SALARY_FACTOR
    + AVAILABILITY_FACTOR
)


def map_sentiment_score(sentiment: str) -> float:
    return SENTIMENT_WEIGHTS[sentiment]


def map_confidence_score(confidence: str) -> float:
    return CONFIDENCE_WEIGHTS[confidence]


def map_specificity_score(specificity: str) -> float:
    return SPECIFICITY_WEIGHTS[specificity]


def map_salary_match_score(salary_alignment: str) -> float:
    return SALARY_ALIGNMENT_WEIGHTS[salary_alignment]


def map_availability_score(availability_days: int | None) -> float:
    if availability_days is None:
        return 0.5
    if availability_days <= 30:
        return 1.0
    if availability_days <= 60:
        return 0.5
    return 0.0


def build_interest_explanation(
    conversation: CandidateConversation,
    breakdown: InterestScoreBreakdown,
    interest_score: float,
) -> str:
    signals = conversation.signals
    availability_text = (
        f"{signals.availability_days} days" if signals.availability_days is not None else "unknown"
    )

    return (
        f"Sentiment {signals.sentiment} ({breakdown.sentiment_score:.1f}), "
        f"confidence {signals.confidence} ({breakdown.confidence_score:.1f}), "
        f"specificity {signals.specificity} ({breakdown.specificity_score:.1f}), "
        f"salary alignment {signals.salary_alignment} ({breakdown.salary_match_score:.1f}), "
        f"availability {availability_text} ({breakdown.availability_score:.1f}). "
        f"Final Interest Score {interest_score:.1f}%."
    )


def score_candidate_interest(conversation: CandidateConversation) -> CandidateInterestResult:
    signals = conversation.signals

    if not signals.consent_given:
        breakdown = InterestScoreBreakdown(
            sentiment_score=0.0,
            confidence_score=0.0,
            specificity_score=0.0,
            salary_match_score=0.0,
            availability_score=0.0,
        )
        explanation = (
            "Candidate did not consent to continue the conversation, so the Interest Score is 0.0%."
        )
        return CandidateInterestResult(
            candidate_id=conversation.candidate_id,
            full_name=conversation.full_name,
            role_title=conversation.role_title,
            interest_score=0.0,
            breakdown=breakdown,
            explanation=explanation,
            conversation_id=conversation.conversation_id,
            provider=conversation.provider,
        )

    breakdown = InterestScoreBreakdown(
        sentiment_score=map_sentiment_score(signals.sentiment),
        confidence_score=map_confidence_score(signals.confidence),
        specificity_score=map_specificity_score(signals.specificity),
        salary_match_score=map_salary_match_score(signals.salary_alignment),
        availability_score=map_availability_score(signals.availability_days),
    )

    weighted_score = (
        (SENTIMENT_FACTOR * breakdown.sentiment_score)
        + (CONFIDENCE_FACTOR * breakdown.confidence_score)
        + (SPECIFICITY_FACTOR * breakdown.specificity_score)
        + (SALARY_FACTOR * breakdown.salary_match_score)
        + (AVAILABILITY_FACTOR * breakdown.availability_score)
    )
    interest_score = (weighted_score / TOTAL_FACTOR) * 100

    return CandidateInterestResult(
        candidate_id=conversation.candidate_id,
        full_name=conversation.full_name,
        role_title=conversation.role_title,
        interest_score=round(interest_score, 2),
        breakdown=breakdown,
        explanation=build_interest_explanation(conversation, breakdown, interest_score),
        conversation_id=conversation.conversation_id,
        provider=conversation.provider,
    )
