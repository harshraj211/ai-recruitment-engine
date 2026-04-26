"""Quick test: POST /api/v1/match and print result summary."""
import time
import requests

url = "http://127.0.0.1:8009/api/v1/match"
payload = {
    "job_description": (
        "We are hiring a Senior Machine Learning Engineer. "
        "Must have Python, PyTorch, Docker, AWS, and vector search. "
        "Salary budget 50000 to 65000 USD. Remote."
    ),
    "top_k_final": 3,
}

t = time.time()
r = requests.post(url, json=payload, timeout=120)
elapsed = time.time() - t

print(f"Status: {r.status_code} in {elapsed:.1f}s")
if r.status_code == 200:
    d = r.json()
    print(f"Rankings: {len(d.get('rankings', []))} candidates")
    for c in d.get("rankings", []):
        print(f"  #{c['rank']} {c['candidate_name']} — Final: {c['final_score']}")
else:
    print(r.text[:500])
