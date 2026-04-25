from app.services.candidate_store import load_candidate_lookup
from app.services.interest_scoring import (
    PredictiveEngagementService,
    calculate_salary_alignment,
    score_availability,
    score_salary_alignment,
    score_tenure_peak,
)
from app.services.jd_parser import parse_job_description


def test_score_salary_alignment_prefers_in_range_or_below_budget() -> None:
    assert score_salary_alignment("aligned") > score_salary_alignment("above_range")
    assert score_salary_alignment("below_range") == 1.0
    assert score_salary_alignment("unknown") == 0.5


def test_score_availability_prefers_faster_candidates() -> None:
    assert score_availability(10) > score_availability(45)
    assert score_availability(30) > score_availability(60)


def test_score_tenure_peak_favors_move_windows() -> None:
    assert score_tenure_peak(1.5) > score_tenure_peak(0.4)
    assert score_tenure_peak(3.0) > score_tenure_peak(6.0)


def test_predictive_engagement_returns_explainable_scores() -> None:
    candidate = load_candidate_lookup()["cand-002"]
    parsed_jd = parse_job_description(
        """
        We are hiring a Senior Machine Learning Engineer.
        Must have Python, FastAPI, PyTorch, Docker, AWS, and vector search.
        Nice to have MLflow.
        Budget: $50,000 - $65,000 annually.
        """
    )

    result = PredictiveEngagementService().score_candidate(candidate, parsed_jd)

    assert result.interest_score > 0
    assert result.flight_risk_score > 0
    assert result.salary_alignment == "aligned"
    assert "Flight risk" in result.explanation


def test_salary_alignment_utility_matches_expected_bands() -> None:
    assert calculate_salary_alignment(55000, [50000, 65000]) == "aligned"
    assert calculate_salary_alignment(45000, [50000, 65000]) == "below_range"
    assert calculate_salary_alignment(70000, [50000, 65000]) == "above_range"
