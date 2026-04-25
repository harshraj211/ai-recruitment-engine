from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.candidate_store import load_candidates


def main() -> None:
    candidates = load_candidates()
    print(f"Loaded {len(candidates)} candidates")
    print()

    for candidate in candidates[:3]:
        print(
            {
                "id": candidate.id,
                "full_name": candidate.full_name,
                "role_title": candidate.role_title,
                "experience_years": candidate.total_experience_years,
                "skills": candidate.skills[:5],
                "expected_salary_usd": candidate.expected_salary_usd,
                "availability_days": candidate.availability_days,
            }
        )


if __name__ == "__main__":
    main()
