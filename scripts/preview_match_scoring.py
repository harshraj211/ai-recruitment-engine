from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.candidate_store import load_candidates
from app.services.jd_parser import parse_job_description
from app.services.match_scoring import rank_candidates_by_match

SAMPLE_JOB_DESCRIPTION = """
We are hiring a Senior Machine Learning Engineer for our talent intelligence platform.
You should have 4+ years of experience building production APIs and ML services.
Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search.
Budget: $50,000 - $65,000 annually.
This is a remote role.
"""


def main() -> None:
    parsed_jd = parse_job_description(SAMPLE_JOB_DESCRIPTION)
    candidates = load_candidates()
    ranked_matches = rank_candidates_by_match(parsed_jd, candidates)[:5]

    print("PARSED JD")
    print(parsed_jd.model_dump())
    print()
    print("TOP MATCH SCORES")

    for result in ranked_matches:
        print(
            {
                "candidate_id": result.candidate_id,
                "full_name": result.full_name,
                "role_title": result.role_title,
                "match_score": result.match_score,
                "skill_match_score": result.skill_match_score,
                "experience_match_score": result.experience_match_score,
                "matched_skills": result.matched_skills,
                "missing_skills": result.missing_skills,
            }
        )
        print(result.explanation)
        print()


if __name__ == "__main__":
    main()
