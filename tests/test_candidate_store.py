from app.services.candidate_store import build_candidate_search_text, load_candidates


def test_candidate_dataset_loads_with_expected_size() -> None:
    candidates = load_candidates()
    assert 20 <= len(candidates) <= 50
    assert len({candidate.id for candidate in candidates}) == len(candidates)


def test_candidate_records_include_step_2_fields() -> None:
    candidate = load_candidates()[0]
    assert candidate.skills
    assert candidate.total_experience_years > 0
    assert candidate.expected_salary_usd > 0
    assert candidate.availability_days >= 0


def test_candidate_search_text_contains_core_profile_information() -> None:
    candidate = load_candidates()[0]
    search_text = build_candidate_search_text(candidate)
    assert candidate.role_title in search_text
    assert candidate.profile_summary in search_text
    assert "Skills:" in search_text
