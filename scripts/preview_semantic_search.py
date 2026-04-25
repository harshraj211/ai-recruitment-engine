from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.jd_parser import parse_job_description
from app.services.vector_store import CandidateVectorStore

SAMPLE_JOB_DESCRIPTION = """
We are hiring a Senior Machine Learning Engineer for our talent intelligence platform.
You should have 4+ years of experience building production APIs and ML services.
Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search.
Budget: $50,000 - $65,000 annually.
This is a remote role.
"""


def main() -> None:
    parsed_jd = parse_job_description(SAMPLE_JOB_DESCRIPTION)
    store = CandidateVectorStore()
    build_summary = store.build_index()
    results = store.search_parsed_job(parsed_jd, top_k=5)

    print("INDEX")
    print(build_summary)
    print()
    print("QUERY")
    print(parsed_jd.model_dump())
    print()
    print("RESULTS")
    for result in results:
        print(
            {
                "candidate_id": result.candidate_id,
                "full_name": result.full_name,
                "role_title": result.role_title,
                "similarity_score": round(result.similarity_score, 4),
                "experience_years": result.total_experience_years,
                "top_skills": result.skills[:5],
            }
        )


if __name__ == "__main__":
    main()
