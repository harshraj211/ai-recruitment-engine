from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.final_ranking import FinalRankingService
from app.services.jd_parser import parse_job_description

SAMPLE_JOB_DESCRIPTION = """
We are hiring a Senior Machine Learning Engineer for our talent intelligence platform.
You should have 4+ years of experience building production APIs and ML services.
Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search.
Budget: $50,000 - $65,000 annually.
This is a remote role.
"""


def main() -> None:
    parsed_jd = parse_job_description(SAMPLE_JOB_DESCRIPTION)
    rankings = FinalRankingService().rank_candidates(parsed_jd, top_k_search=5, top_k_final=5)

    print("FINAL RANKING")
    for result in rankings:
        print(
            {
                "rank": result.rank,
                "candidate_id": result.candidate_id,
                "full_name": result.full_name,
                "role_title": result.role_title,
                "final_score": result.final_score,
                "match_score": result.match_result.match_score,
                "interest_score": result.interest_result.interest_score,
            }
        )
        print(result.final_explanation)
        print()


if __name__ == "__main__":
    main()
