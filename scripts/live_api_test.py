"""Live end-to-end API test for the upgraded talent intelligence pipeline."""

import httpx
import os

BASE_URL = os.environ.get("TALENT_SCOUT_URL", "http://127.0.0.1:8000")

response = httpx.post(
    f"{BASE_URL}/api/v1/match",
    json={
        "job_description": (
            "We are hiring a Senior Machine Learning Engineer for our AI-powered "
            "talent intelligence platform. Must have Python, FastAPI, PyTorch, Docker, "
            "AWS, MLflow, and vector search. Nice to have RAG and Kubernetes. "
            "Experience in HR tech or search is preferred. This is a remote position "
            "with a salary budget of $50,000 to $65,000 USD annually."
        ),
        "top_k_search": 10,
        "top_k_final": 5,
        "page": 1,
        "page_size": 5,
        "include_outreach": True,
    },
    timeout=120,
)

print("STATUS:", response.status_code)
data = response.json()

parsed = data["parsed_job_description"]
print("\n=== PARSED JOB DESCRIPTION ===")
print(f"  Role: {parsed.get('role_title')}")
print(f"  Seniority: {parsed.get('seniority')}")
print(f"  Min Experience: {parsed.get('min_experience_years')} years")
print(f"  Mandatory Skills: {', '.join(parsed.get('mandatory_skills', []))}")
print(f"  Nice To Have: {', '.join(parsed.get('nice_to_have_skills', []))}")
print(f"  Domain Knowledge: {', '.join(parsed.get('domain_knowledge', []))}")
print(f"  Salary Range: {parsed.get('salary_range_usd')}")
print(f"  Work Mode: {parsed.get('work_mode')}")

print(f"\n=== TOP {len(data['rankings'])} CANDIDATES ===")
print(f"  Retrieved: {data['total_candidates_retrieved']}")
print(f"  Considered: {data['total_candidates_considered']}")
print(f"  Returned: {data['total_candidates_returned']}")
print(f"  Page: {data['page']} / {data['total_pages']}")

for ranking in data["rankings"]:
    print(f"\n--- Rank {ranking['rank']}: {ranking['candidate_name']} ({ranking['candidate_id']}) ---")
    print(f"  Role: {ranking['role_title']}")
    print(f"  Final Score: {ranking['final_score']}")
    print(f"  Match Score: {ranking['match_score']}")
    print(f"  Interest Score: {ranking['interest_score']}")
    print(f"  BM25 Score: {ranking['bm25_score']}")
    print(f"  Cross Encoder Score: {ranking['cross_encoder_score']}")
    print(f"  Flight Risk Score: {ranking['flight_risk_score']}")
    print(f"  Summary: {ranking['summary']}")
    print(f"  Missing Skills: {', '.join(ranking['missing_skills'])}")
    print(f"  Recommendation: {ranking['recommendation']}")
    if ranking.get("recruiter_outreach"):
        print(f"  Outreach: {ranking['recruiter_outreach']['message']}")
