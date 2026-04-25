from app.schemas.final_ranking import FinalCandidateRanking, RecruiterDecisionBrief

DEFAULT_MATCH_WEIGHT = 0.50
DEFAULT_INTEREST_WEIGHT = 0.25
DEFAULT_CROSS_ENCODER_WEIGHT = 0.25


def calculate_final_score(
    match_score: float,
    interest_score: float,
    cross_encoder_score: float,
) -> float:
    return (
        (DEFAULT_MATCH_WEIGHT * match_score)
        + (DEFAULT_INTEREST_WEIGHT * interest_score)
        + (DEFAULT_CROSS_ENCODER_WEIGHT * cross_encoder_score)
    )


def build_final_explanation(
    final_score: float,
    match_result,
    interest_result,
    cross_encoder_score: float,
) -> str:
    return (
        f"Final Score {final_score:.1f}% combines Technical Match {match_result.match_score:.1f}%, "
        f"Interest {interest_result.interest_score:.1f}%, and Re-ranker {cross_encoder_score:.1f}%. "
        f"{match_result.explanation} {interest_result.explanation}"
    )


def build_skill_match_reason(match_result) -> str:
    details = "; ".join(match_result.skill_alignment_details[:4])
    return (
        f"Weighted skill coverage {match_result.skill_match_score * 100:.1f}% with "
        f"mandatory coverage {match_result.core_skill_score * 100:.1f}% and "
        f"nice-to-have coverage {match_result.secondary_skill_score * 100:.1f}%. "
        f"{details}"
    ).strip()


def build_experience_match_reason(match_result, candidate) -> str:
    return (
        f"Experience fit {match_result.experience_match_score * 100:.1f}% with "
        f"{candidate.total_experience_years:.1f} years in {candidate.role_title}. "
        f"Trajectory boost {match_result.trajectory_boost_score * 100:.1f}%."
    )


def build_interest_insight(interest_result) -> str:
    return (
        f"Interest Score {interest_result.interest_score:.1f}% based on salary alignment, "
        f"availability, and engagement probability. Flight risk is {interest_result.flight_risk_score:.1f}%."
    )


def build_salary_alignment_reason(interest_result) -> str:
    return f"Salary alignment is {interest_result.salary_alignment}."


def build_availability_insight(interest_result) -> str:
    availability = interest_result.availability_days
    return f"Availability is {availability} days." if availability is not None else "Availability is unknown."


def build_recommendation(final_score: float, match_result, interest_result) -> str:
    if match_result.missing_core_skills:
        return "Review manually before outreach due to missing critical skills."
    if final_score >= 80 and interest_result.flight_risk_score >= 55:
        return "Prioritize outreach this week."
    if final_score >= 70:
        return "Advance to recruiter screen."
    if final_score >= 60:
        return "Keep warm for secondary review."
    return "Do not prioritize for this requisition."


def build_decision_brief(
    final_score: float,
    match_result,
    interest_result,
    parsed_jd,
    candidate,
) -> RecruiterDecisionBrief:
    missing_core = match_result.missing_core_skills
    matched_core = match_result.matched_core_skills or match_result.matched_skills
    role_title = parsed_jd.role_title or "the target role"

    if missing_core:
        verdict = "Manual review"
        next_action = "Validate the missing mandatory skill evidence before outreach."
        risk_to_watch = f"Mandatory gap: {missing_core[0]}."
    elif final_score >= 82 and interest_result.interest_score >= 75:
        verdict = "Fast-track"
        next_action = "Prioritize outreach and move to recruiter screen."
        risk_to_watch = "Keep the first touch specific so the candidate sees role relevance quickly."
    elif final_score >= 70:
        verdict = "Recruiter screen"
        next_action = "Advance after confirming interest and availability."
        risk_to_watch = "Interest or re-ranker signal is solid but not decisive."
    else:
        verdict = "Keep warm"
        next_action = "Save for a secondary pass or a broader requisition."
        risk_to_watch = "Overall fit is below the current shortlist threshold."

    strongest_skill = matched_core[0] if matched_core else candidate.role_title
    outreach_angle = (
        f"Lead with {strongest_skill} and connect it directly to the {role_title} scope."
    )

    if interest_result.salary_alignment == "above_range":
        negotiation_note = "Candidate expectation is above budget; confirm flexibility early."
    elif interest_result.salary_alignment == "below_range":
        negotiation_note = "Compensation appears favorable; focus on scope and growth."
    elif interest_result.salary_alignment == "aligned":
        negotiation_note = "Salary expectation is inside range; avoid delaying on compensation."
    else:
        negotiation_note = "Salary range is unknown; ask early to avoid late-stage mismatch."

    evidence = [
        f"Match {match_result.match_score:.1f}%",
        f"Interest {interest_result.interest_score:.1f}%",
        f"Availability {interest_result.availability_days} days"
        if interest_result.availability_days is not None
        else "Availability unknown",
    ]
    if matched_core:
        evidence.append(f"Core evidence: {', '.join(matched_core[:3])}")
    if missing_core:
        evidence.append(f"Missing mandatory: {', '.join(missing_core[:2])}")

    return RecruiterDecisionBrief(
        verdict=verdict,
        next_action=next_action,
        outreach_angle=outreach_angle,
        risk_to_watch=risk_to_watch,
        negotiation_note=negotiation_note,
        evidence=evidence,
    )


def candidate_ranking_sort_key(item: FinalCandidateRanking) -> tuple:
    return (
        0 if item.match_result.missing_core_skills else -1,
        -item.match_result.core_skill_score,
        -item.final_score,
        -item.match_result.match_score,
        -item.interest_result.interest_score,
        -(item.cross_encoder_score or 0.0),
        len(item.match_result.missing_core_skills),
        item.candidate_id,
    )
