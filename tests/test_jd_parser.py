from app.services.jd_parser import parse_job_description


def test_parser_extracts_role_skills_experience_salary_and_work_mode() -> None:
    jd_text = """
    We are hiring a Senior Machine Learning Engineer for our AI team.
    Candidates should have 4+ years of experience with Python, FastAPI, PyTorch,
    Docker, AWS, MLflow, and vector search systems.
    Budget: $50,000 - $65,000 annually.
    This is a remote role.
    """

    parsed = parse_job_description(jd_text)

    assert parsed.role_title == "Machine Learning Engineer"
    assert parsed.seniority == "senior"
    assert parsed.min_experience_years == 4
    assert parsed.salary_range_usd == [50000, 65000]
    assert parsed.work_mode == "remote"
    assert parsed.skills[:7] == [
        "Python",
        "FastAPI",
        "PyTorch",
        "Docker",
        "AWS",
        "MLflow",
        "Vector Search",
    ]


def test_parser_handles_skill_synonyms_and_experience_ranges() -> None:
    jd_text = """
    Looking for an NLP Engineer with 3-5 years of experience in sentence transformers,
    scikit learn, spaCy, information extraction, and Python.
    """

    parsed = parse_job_description(jd_text)

    assert parsed.role_title == "NLP Engineer"
    assert parsed.min_experience_years == 3
    assert parsed.skills == [
        "SentenceTransformers",
        "Scikit-learn",
        "spaCy",
        "Information Extraction",
        "Python",
    ]


def test_parser_can_infer_experience_from_seniority_when_years_are_missing() -> None:
    jd_text = """
    We need a lead backend engineer to build scalable REST APIs with Python,
    PostgreSQL, Kafka, and Redis. Hybrid work setup.
    """

    parsed = parse_job_description(jd_text)

    assert parsed.role_title == "Backend Engineer"
    assert parsed.seniority == "lead"
    assert parsed.min_experience_years == 8
    assert parsed.work_mode == "hybrid"
    assert parsed.skills == ["REST APIs", "Python", "PostgreSQL", "Kafka", "Redis"]
