from __future__ import annotations

import math
from datetime import date

from app.schemas.candidate import Candidate


def years_between(start_date: date, end_date: date | None = None) -> float:
    end = end_date or date.today()
    return max((end - start_date).days / 365.25, 0.0)


def recency_weight(end_date: date | None) -> float:
    if end_date is None:
        return 1.0

    age_years = years_between(end_date, date.today())
    return max(0.35, math.exp(-0.18 * age_years))


def build_skill_recency_weights(candidate: Candidate) -> dict[str, float]:
    weights: dict[str, float] = {}

    for skill in candidate.skills:
        weights[skill] = max(weights.get(skill, 0.0), 1.0)

    for entry in candidate.role_history:
        weight = recency_weight(entry.end_date)
        for skill in entry.skills:
            weights[skill] = max(weights.get(skill, 0.0), weight)

    return weights


def latest_tenure_years(candidate: Candidate) -> float:
    if not candidate.role_history:
        return candidate.total_experience_years
    latest = candidate.role_history[-1]
    return years_between(latest.start_date, latest.end_date)


def stagnation_score(candidate: Candidate) -> float:
    if len(candidate.role_history) <= 1:
        return min(latest_tenure_years(candidate) / 4.0, 1.0)

    latest_title = candidate.role_history[-1].title.lower()
    prior_titles = {entry.title.lower() for entry in candidate.role_history[:-1]}
    repeated_title = latest_title in prior_titles
    tenure = latest_tenure_years(candidate)
    score = 0.2 + min(tenure / 5.0, 0.8)
    if repeated_title:
        score = min(score + 0.15, 1.0)
    return score


def promotion_velocity(candidate: Candidate) -> float:
    if len(candidate.role_history) <= 1:
        return 0.0

    promotions = 0
    compressed_years = 0.0
    for previous, current in zip(candidate.role_history, candidate.role_history[1:], strict=False):
        if previous.title.lower() != current.title.lower():
            promotions += 1
            compressed_years += years_between(previous.start_date, previous.end_date or current.start_date)

    if promotions == 0:
        return 0.0

    average_years = compressed_years / promotions if promotions else 0.0
    if average_years <= 2.0:
        return 1.0
    if average_years <= 3.0:
        return 0.75
    if average_years <= 4.0:
        return 0.45
    return 0.2


def career_trajectory_boost(candidate: Candidate) -> float:
    velocity = promotion_velocity(candidate)
    if velocity >= 0.95:
        return 0.10
    if velocity >= 0.70:
        return 0.07
    if velocity >= 0.40:
        return 0.05
    return 0.0
