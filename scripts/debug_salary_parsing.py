"""Debug salary parsing for the demo JD."""
from app.services.jd_parser import parse_job_description, extract_salary_range_usd

jd = (
    "We are hiring a Senior Machine Learning Engineer for our AI-powered "
    "talent intelligence platform. The ideal candidate should have 4+ years "
    "of experience building production-grade ML services and APIs. Required "
    "skills include Python, FastAPI, PyTorch, Docker, AWS, MLflow, Machine "
    "Learning, and Vector Search. This is a remote position with a salary "
    "budget of $50,000 to $65,000 USD annually. You will work on "
    "recommendation engines, model serving pipelines, and semantic search "
    "infrastructure."
)

# Test extract_salary_range_usd directly
salary = extract_salary_range_usd(jd.lower())
print("Salary extraction:", salary)

# Test full parse
parsed = parse_job_description(jd)
print("Parsed salary_range_usd:", parsed.salary_range_usd)
print("Parsed role_title:", parsed.role_title)
print("Parsed skills:", parsed.skills)
print("Parsed min_experience_years:", parsed.min_experience_years)
print("Parsed work_mode:", parsed.work_mode)
