from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.candidate_store import load_candidate_lookup
from app.services.conversation_service import ConversationService
from app.services.interest_scoring import score_candidate_interest
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
    candidate = load_candidate_lookup()["cand-002"]
    conversation = ConversationService().simulate_conversation(candidate, parsed_jd)
    interest_result = score_candidate_interest(conversation)

    print("CONVERSATION")
    print(
        {
            "conversation_id": conversation.conversation_id,
            "provider": conversation.provider,
            "signals": conversation.signals.model_dump(),
        }
    )
    print()
    print("INTEREST SCORE")
    print(
        {
            "candidate_id": interest_result.candidate_id,
            "full_name": interest_result.full_name,
            "interest_score": interest_result.interest_score,
            "breakdown": interest_result.breakdown.model_dump(),
        }
    )
    print(interest_result.explanation)


if __name__ == "__main__":
    main()
