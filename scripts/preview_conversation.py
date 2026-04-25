from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.candidate_store import load_candidate_lookup
from app.services.conversation_service import ConversationService
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
    service = ConversationService()
    conversation = service.simulate_conversation(candidate, parsed_jd)

    print("MODE")
    print({"provider": conversation.provider, "model": conversation.model})
    print()
    print("SIGNALS")
    print(conversation.signals.model_dump())
    print()
    print("TRANSCRIPT")
    for turn in conversation.transcript:
        print(f"[{turn.stage}] {turn.speaker}: {turn.message}")
    print()
    print("SUMMARY")
    print(conversation.summary)
    print()
    print("STORED_AT")
    print(conversation.storage_path)


if __name__ == "__main__":
    main()
