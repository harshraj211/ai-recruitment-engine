from app.services.jd_parser import parse_job_description


def test_parser_extracts_taxonomy_experience_salary_and_work_mode() -> None:
    jd_text = """
    We are hiring a Senior Machine Learning Engineer for our AI team.
    Must have Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search.
    Nice to have RAG and Kubernetes.
    Experience in HR tech or search platforms is preferred.
    Budget: $50,000 - $65,000 annually.
    This is a remote role.
    """

    parsed = parse_job_description(jd_text)

    assert parsed.role_title == "Machine Learning Engineer"
    assert parsed.seniority == "senior"
    assert parsed.min_experience_years == 5
    assert parsed.salary_range_usd == [50000, 65000]
    assert parsed.work_mode == "remote"
    assert "Python" in parsed.mandatory_skills
    assert "Vector Search" in parsed.mandatory_skills
    assert "RAG" in parsed.nice_to_have_skills
    assert "HR Tech" in parsed.domain_knowledge or "Search" in parsed.domain_knowledge


def test_parser_handles_skill_synonyms_and_experience_ranges() -> None:
    jd_text = """
    Looking for an NLP Engineer with 3-5 years of experience in sentence transformers,
    scikit learn, spaCy, information extraction, and Python.
    """

    parsed = parse_job_description(jd_text)

    assert parsed.role_title == "NLP Engineer"
    assert parsed.min_experience_years == 3
    assert "SentenceTransformers" in parsed.skills
    assert "Scikit-learn" in parsed.skills
    assert "spaCy" in parsed.skills
    assert "Information Extraction" in parsed.skills
    assert "Python" in parsed.skills


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
    assert "REST APIs" in parsed.skills
    assert "Python" in parsed.skills
