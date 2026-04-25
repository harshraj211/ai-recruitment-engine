from app.services.candidate_store import load_candidates
from app.services.jd_parser import parse_job_description
from app.services.match_scoring import (
    calculate_experience_match,
    calculate_skill_match,
    rank_candidates_by_match,
    score_candidate_match,
)


def test_calculate_skill_match_returns_overlap_and_missing_skills() -> None:
    candidate = load_candidates()[1]
    parsed_jd = parse_job_description(
        """
        We are hiring a Machine Learning Engineer with Python, FastAPI, PyTorch,
        Docker, AWS, MLflow, and vector search experience.
        """
    )

    score, matched_skills, missing_skills = calculate_skill_match(parsed_jd, candidate)

    assert score == 6 / 7
    assert matched_skills == ["Python", "FastAPI", "PyTorch", "Docker", "AWS", "MLflow"]
    assert missing_skills == ["Vector Search"]


def test_calculate_experience_match_caps_at_one() -> None:
    candidate = load_candidates()[2]
    parsed_jd = parse_job_description("Need a backend engineer with 4+ years of experience.")

    experience_score = calculate_experience_match(parsed_jd, candidate)

    assert experience_score == 1.0


def test_score_candidate_match_uses_srs_weighted_formula() -> None:
    candidate = load_candidates()[1]
    parsed_jd = parse_job_description(
        """
        We are hiring a Machine Learning Engineer with 4+ years of experience.
        Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search.
        """
    )

    result = score_candidate_match(parsed_jd, candidate)

    assert result.skill_match_score == 6 / 7
    assert result.experience_match_score == 1.0
    assert result.match_score == 90.0
    assert "Final Match Score 90.0%" in result.explanation


def test_rank_candidates_by_match_returns_highest_scores_first() -> None:
    parsed_jd = parse_job_description(
        """
        We are hiring a Senior Machine Learning Engineer with 4+ years of experience.
        Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search.
        """
    )

    ranked_results = rank_candidates_by_match(parsed_jd, load_candidates())

    assert ranked_results[0].candidate_id == "cand-002"
    assert ranked_results[0].match_score >= ranked_results[1].match_score
    assert ranked_results[0].matched_skills == [
        "Python",
        "FastAPI",
        "PyTorch",
        "Docker",
        "AWS",
        "MLflow",
    ]
