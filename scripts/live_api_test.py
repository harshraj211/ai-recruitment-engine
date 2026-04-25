"""Live end-to-end API test with proper dollar signs in salary."""
import httpx
import json

response = httpx.post(
    "http://127.0.0.1:8811/api/v1/match",
    json={
        "job_description": (
            "We are hiring a Senior Machine Learning Engineer for our AI-powered "
            "talent intelligence platform. The ideal candidate should have 4+ years "
            "of experience building production-grade ML services and APIs. Required "
            "skills include Python, FastAPI, PyTorch, Docker, AWS, MLflow, Machine "
            "Learning, and Vector Search. This is a remote position with a salary "
            "budget of $50,000 to $65,000 USD annually. You will work on "
            "recommendation engines, model serving pipelines, and semantic search "
            "infrastructure."
        ),
        "top_k_search": 5,
        "top_k_final": 3,
    },
    timeout=120,
)

print("STATUS:", response.status_code)
data = response.json()

# Show parsed JD
pjd = data["parsed_job_description"]
print("\n=== PARSED JOB DESCRIPTION ===")
print(f"  Role: {pjd['role_title']}")
print(f"  Seniority: {pjd['seniority']}")
print(f"  Min Experience: {pjd['min_experience_years']} years")
print(f"  Skills: {', '.join(pjd['skills'])}")
print(f"  Salary Range: {pjd['salary_range_usd']}")
print(f"  Work Mode: {pjd['work_mode']}")

# Show rankings
print(f"\n=== TOP {len(data['rankings'])} CANDIDATES ===")
print(f"  Retrieved: {data['total_candidates_retrieved']}")
print(f"  Considered: {data['total_candidates_considered']}")
print(f"  Returned: {data['total_candidates_returned']}")

for r in data["rankings"]:
    print(f"\n--- Rank {r['rank']}: {r['full_name']} ({r['candidate_id']}) ---")
    print(f"  Role: {r['role_title']}")
    print(f"  Final Score: {r['final_score']}")
    print(f"  Match Score: {r['match_result']['match_score']}")
    print(f"  Interest Score: {r['interest_result']['interest_score']}")
    print(f"  Skill Match: {r['skill_match_reason']}")
    print(f"  Experience: {r['experience_match_reason']}")
    print(f"  Conversation: {r['conversation_insight']}")
    print(f"  Provider: {r['interest_result']['provider']}")
