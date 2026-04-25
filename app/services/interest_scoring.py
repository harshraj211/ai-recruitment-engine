import asyncio
import math

from app.schemas.candidate import Candidate
from app.schemas.interest_scoring import CandidateInterestResult, InterestScoreBreakdown
from app.schemas.job_description import ParsedJobDescription
from app.services.experience_intelligence import latest_tenure_years, promotion_velocity, stagnation_score
from app.services.match_scoring import calculate_role_alignment

SALARY_ALIGNMENT_WEIGHTS = {
    "aligned": 0.85,
    "below_range": 1.0,
    "above_range": 0.15,
    "unknown": 0.50,
}

ENGAGEMENT_STATUS_WEIGHTS = {
    "open_to_work": 1.0,
    "exploring": 0.78,
    "passive": 0.52,
}


def calculate_salary_alignment(
    salary_expectation_usd: int | None,
    salary_range_usd: list[int],
) -> str:
    if salary_expectation_usd is None or len(salary_range_usd) != 2:
        return "unknown"

    low, high = sorted(salary_range_usd)
    if low <= salary_expectation_usd <= high:
        return "aligned"
    if salary_expectation_usd < low:
        return "below_range"
    return "above_range"


def score_salary_alignment(salary_alignment: str) -> float:
    return SALARY_ALIGNMENT_WEIGHTS[salary_alignment]


def score_availability(availability_days: int | None) -> float:
    if availability_days is None:
        return 0.5
    if availability_days <= 15:
        return 1.0
    if availability_days <= 30:
        return 0.85
    if availability_days <= 45:
        return 0.65
    if availability_days <= 60:
        return 0.45
    return 0.2


def score_tenure_peak(tenure_years: float) -> float:
    peaks = (1.5, 3.0, 4.0)
    best = 0.0
    for peak in peaks:
        best = max(best, math.exp(-((tenure_years - peak) ** 2) / (2 * 0.85**2)))
    return min(best, 1.0)


def score_engagement_probability(candidate: Candidate) -> float:
    base = ENGAGEMENT_STATUS_WEIGHTS.get(candidate.current_status, 0.55)
    if candidate.work_preference == "remote":
        base = min(base + 0.05, 1.0)
    return base


def build_interest_explanation(
    salary_alignment: str,
    breakdown: InterestScoreBreakdown,
    interest_score: float,
    flight_risk_score: float,
    availability_days: int | None,
) -> str:
    availability_text = f"{availability_days} days" if availability_days is not None else "unknown"
    return (
        f"Predictive engagement uses tenure peak {breakdown.tenure_score:.2f}, "
        f"salary alignment {salary_alignment} ({breakdown.salary_alignment_score:.2f}), "
        f"availability {availability_text} ({breakdown.availability_score:.2f}), "
        f"career stagnation ({breakdown.stagnation_score:.2f}), "
        f"promotion likelihood ({breakdown.promotion_likelihood_score:.2f}), "
        f"role relevance ({breakdown.role_relevance_score:.2f}), "
        f"engagement probability ({breakdown.engagement_probability_score:.2f}). "
        f"Flight risk {flight_risk_score:.1f}%. Final Interest Score {interest_score:.1f}%."
    )


class PredictiveEngagementService:
    model_version = "predicted-interest-v2"

    def score_candidate(
        self,
        candidate: Candidate,
        parsed_jd: ParsedJobDescription,
    ) -> CandidateInterestResult:
        salary_alignment = calculate_salary_alignment(
            candidate.expected_salary_usd,
            parsed_jd.salary_range_usd,
        )
        tenure = latest_tenure_years(candidate)
        promotion_score = promotion_velocity(candidate)
        breakdown = InterestScoreBreakdown(
            tenure_score=score_tenure_peak(tenure),
            salary_alignment_score=score_salary_alignment(salary_alignment),
            availability_score=score_availability(candidate.availability_days),
            stagnation_score=stagnation_score(candidate),
            promotion_likelihood_score=promotion_score,
            role_relevance_score=calculate_role_alignment(parsed_jd, candidate),
            engagement_probability_score=score_engagement_probability(candidate),
        )

        flight_risk = (
            (0.32 * breakdown.tenure_score)
            + (0.28 * breakdown.salary_alignment_score)
            + (0.25 * breakdown.stagnation_score)
            + (0.15 * (1.0 - breakdown.promotion_likelihood_score))
        ) * 100

        interest_score = (
            (0.35 * (flight_risk / 100.0))
            + (0.20 * breakdown.role_relevance_score)
            + (0.15 * breakdown.availability_score)
            + (0.15 * breakdown.engagement_probability_score)
            + (0.15 * breakdown.salary_alignment_score)
        ) * 100

        return CandidateInterestResult(
            candidate_id=candidate.id,
            full_name=candidate.full_name,
            role_title=candidate.role_title,
            interest_score=round(interest_score, 2),
            flight_risk_score=round(flight_risk, 2),
            breakdown=breakdown,
            salary_alignment=salary_alignment,
            availability_days=candidate.availability_days,
            model_version=self.model_version,
            explanation=build_interest_explanation(
                salary_alignment,
                breakdown,
                interest_score,
                flight_risk,
                candidate.availability_days,
            ),
            provider="deterministic",
        )

    async def score_candidate_async(
        self,
        candidate: Candidate,
        parsed_jd: ParsedJobDescription,
    ) -> CandidateInterestResult:
        return await asyncio.to_thread(self.score_candidate, candidate, parsed_jd)


def score_candidate_interest(
    candidate: Candidate,
    parsed_jd: ParsedJobDescription,
) -> CandidateInterestResult:
    return PredictiveEngagementService().score_candidate(candidate, parsed_jd)
