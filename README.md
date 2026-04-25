# AI-Powered Talent Scouting & Engagement Agent

Hackathon-ready backend for parsing job descriptions, semantically matching candidates, simulating recruiter outreach, and returning ranked results with explanations.

## Step 1 to Step 9 Status

This repository currently includes:

- Basic FastAPI backend
- Environment-based configuration
- Health and root endpoints
- A realistic local candidate dataset with 20 validated dummy profiles
- Candidate loader utilities for later embedding and FAISS steps
- A rule-based JD parser for role, skills, experience, salary range, and work mode
- SentenceTransformers embedding generation with `all-MiniLM-L6-v2`
- FAISS index building and semantic candidate search
- SRS-style weighted match scoring with explanations
- Groq-ready recruiter-candidate conversation simulation with transcript storage
- Interest scoring based on conversation signals
- Final ranking by combining Match Score and Interest Score
- API endpoint `/api/v1/match` for the full pipeline
- A lightweight pipeline orchestrator that connects all modules end to end
- Ready folders for candidate data, FAISS index files, and conversation logs
- A repo-level `context.md` status tracker

## Recommended Python Version

Use Python 3.11 for the full prototype.

Step 1 works fine on newer Python versions for the API scaffold, but the later `sentence-transformers`, `torch`, and `faiss-cpu` setup is usually smoother on Python 3.11 for a local Windows hackathon environment.

## Project Structure

```text
.
|-- app
|   |-- api
|   |   |-- routes
|   |   |   `-- matching.py
|   |   |   `-- system.py
|   |   `-- router.py
|   |-- core
|   |   `-- config.py
|   |-- schemas
|   |   `-- api.py
|   |   `-- candidate.py
|   |   `-- final_ranking.py
|   |   `-- interest_scoring.py
|   |   `-- pipeline.py
|   |-- services
|   |   `-- candidate_store.py
|   |   `-- conversation_service.py
|   |   `-- embedding_service.py
|   |   `-- final_ranking.py
|   |   `-- interest_scoring.py
|   |   `-- jd_parser.py
|   |   `-- match_scoring.py
|   |   `-- pipeline_service.py
|   |   `-- vector_store.py
|   `-- main.py
|-- data
|   |-- candidates
|   |   `-- candidates.json
|   |-- conversations
|   |-- faiss
|   `-- README.md
|-- scripts
|   `-- build_faiss_index.py
|   `-- preview_candidates.py
|   `-- preview_conversation.py
|   `-- preview_final_ranking.py
|   `-- preview_interest_scoring.py
|   `-- preview_jd_parser.py
|   `-- preview_match_scoring.py
|   `-- preview_semantic_search.py
|-- tests
|   |-- test_candidate_store.py
|   |-- test_conversation_service.py
|   |-- test_final_ranking.py
|   |-- test_interest_scoring.py
|   |-- test_jd_parser.py
|   |-- test_match_api.py
|   |-- test_match_scoring.py
|   |-- test_pipeline_service.py
|   `-- test_system_routes.py
|   `-- test_vector_store.py
|-- context.md
|-- .env.example
|-- .gitignore
`-- requirements.txt
```

## Run Locally

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/api/v1/health`

Preview the local dataset:

```bash
python scripts/preview_candidates.py
```

Preview the JD parser:

```bash
python scripts/preview_jd_parser.py
```

Build the FAISS index:

```bash
py -3.11 scripts/build_faiss_index.py
```

Preview semantic search:

```bash
py -3.11 scripts/preview_semantic_search.py
```

Note: the first embedding run downloads the `all-MiniLM-L6-v2` model from Hugging Face, so it can take a little longer the first time.

Preview match scoring:

```bash
python scripts/preview_match_scoring.py
```

Preview the conversation stage:

```bash
python scripts/preview_conversation.py
```

If `GROQ_API_KEY` is set, the conversation stage uses Groq. If not, the preview runs in a clearly labeled local mock mode so the repository stays runnable.

Preview interest scoring:

```bash
python scripts/preview_interest_scoring.py
```

Preview final ranking:

```bash
py -3.11 scripts/preview_final_ranking.py
```

Run the API:

```bash
py -3.11 uvicorn app.main:app --reload
```

## Example Response

`GET /api/v1/health`

```json
{
  "status": "ok",
  "app_name": "AI-Powered Talent Scouting & Engagement Agent",
  "version": "0.1.0"
}
```

## Candidate Dataset Example

```json
{
  "id": "cand-001",
  "full_name": "Aanya Sharma",
  "role_title": "Senior Data Scientist",
  "experience_years": 7,
  "skills": ["Python", "SQL", "Machine Learning", "XGBoost", "Pandas"],
  "expected_salary_usd": 62000,
  "availability_days": 30
}
```

## JD Parser Example

Input:

```text
We are hiring a Senior Machine Learning Engineer for our talent intelligence platform.
You should have 4+ years of experience building production APIs and ML services.
Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search.
Budget: $50,000 - $65,000 annually.
This is a remote role.
```

Output:

```json
{
  "raw_text": "We are hiring a Senior Machine Learning Engineer for our talent intelligence platform. You should have 4+ years of experience building production APIs and ML services. Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search. Budget: $50,000 - $65,000 annually. This is a remote role.",
  "role_title": "Machine Learning Engineer",
  "seniority": "senior",
  "min_experience_years": 4.0,
  "skills": ["Machine Learning", "Python", "FastAPI", "PyTorch", "Docker", "AWS", "MLflow", "Vector Search"],
  "salary_range_usd": [50000, 65000],
  "work_mode": "remote"
}
```

## Embedding + FAISS Example

Input JD:

```text
We are hiring a Senior Machine Learning Engineer for our talent intelligence platform.
You should have 4+ years of experience building production APIs and ML services.
Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search.
Budget: $50,000 - $65,000 annually.
This is a remote role.
```

Example semantic search output:

```text
INDEX
{'candidate_count': 20, 'embedding_dimension': 384, 'index_path': 'data\\faiss\\candidates.index'}

RESULTS
{'candidate_id': 'cand-002', 'full_name': 'Rohan Mehta', 'role_title': 'Machine Learning Engineer', 'similarity_score': 0.7471, 'experience_years': 5.0, 'top_skills': ['Python', 'FastAPI', 'PyTorch', 'Docker', 'MLflow']}
{'candidate_id': 'cand-015', 'full_name': 'Sneha Patel', 'role_title': 'Computer Vision Engineer', 'similarity_score': 0.6731, 'experience_years': 5.0, 'top_skills': ['Python', 'PyTorch', 'OpenCV', 'Computer Vision', 'Model Optimization']}
{'candidate_id': 'cand-007', 'full_name': 'Meera Iyer', 'role_title': 'MLOps Engineer', 'similarity_score': 0.6655, 'experience_years': 7.0, 'top_skills': ['Python', 'Docker', 'Kubernetes', 'MLflow', 'Airflow']}
```

## Match Scoring Example

Input JD:

```text
We are hiring a Senior Machine Learning Engineer for our talent intelligence platform.
You should have 4+ years of experience building production APIs and ML services.
Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search.
Budget: $50,000 - $65,000 annually.
This is a remote role.
```

Example scoring output:

```text
{'candidate_id': 'cand-002', 'full_name': 'Rohan Mehta', 'role_title': 'Machine Learning Engineer', 'match_score': 82.5, 'skill_match_score': 0.75, 'experience_match_score': 1.0, 'matched_skills': ['Python', 'FastAPI', 'PyTorch', 'Docker', 'AWS', 'MLflow'], 'missing_skills': ['Machine Learning', 'Vector Search']}
Skill match 75.0% (6/8 skills matched): Python, FastAPI, PyTorch, Docker, AWS, MLflow. Missing: Machine Learning, Vector Search. Experience match 100.0% (5.0 yrs vs 4.0 yrs required). Final Match Score 82.5% using skills weight 0.7 and experience weight 0.3.
```

## Conversation Example

Input candidate:

```text
cand-002 | Rohan Mehta | Machine Learning Engineer
```

Example output:

```text
MODE
{'provider': 'mock', 'model': 'mock-local'}

SIGNALS
{'consent_given': True, 'interest_level': 'high', 'sentiment': 'positive', 'confidence': 'high', 'specificity': 'high', 'salary_expectation_usd': 54000, 'salary_alignment': 'aligned', 'availability_days': 21}

TRANSCRIPT
[consent] recruiter: Hi, I'm Talent Scout Bot. I'm reaching out about a Machine Learning Engineer opportunity. Do you have a couple of minutes to chat?
[consent] candidate: Yes, I can spare a few minutes to learn more about the opportunity.
[interest] recruiter: What about this Machine Learning Engineer role sounds interesting to you, and how closely does it match your recent work?
[interest] candidate: Yes, I'm interested because the role lines up well with my recent work in Python, FastAPI, PyTorch. My background as a Machine Learning Engineer feels relevant here.
[salary] recruiter: Our budget for this role is $50,000 to $65,000 USD. Does that align with your expectations?
[salary] candidate: My target is around $54,000 USD, so your range looks workable.
[availability] recruiter: If there is mutual interest, when would you be available to start?
[availability] candidate: I would likely be able to start in about 21 days.
```

## Interest Scoring Example

Example output:

```text
{'candidate_id': 'cand-002', 'full_name': 'Rohan Mehta', 'interest_score': 100.0, 'breakdown': {'sentiment_score': 1.0, 'confidence_score': 1.0, 'specificity_score': 1.0, 'salary_match_score': 1.0, 'availability_score': 1.0}}
Sentiment positive (1.0), confidence high (1.0), specificity high (1.0), salary alignment aligned (1.0), availability 21 days (1.0). Final Interest Score 100.0%.
```

## Final Ranking Example

Example output:

```text
{'rank': 1, 'candidate_id': 'cand-002', 'full_name': 'Rohan Mehta', 'role_title': 'Machine Learning Engineer', 'final_score': 89.5, 'match_score': 82.5, 'interest_score': 100.0}
Final Score 89.5% = (0.6 x Match 82.5) + (0.4 x Interest 100.0).
```

## Match API Example

Request:

```http
POST /api/v1/match
Content-Type: application/json

{
  "job_description": "We are hiring a Senior Machine Learning Engineer for our talent intelligence platform. You should have 4+ years of experience building production APIs and ML services. Required skills: Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search. Budget: $50,000 - $65,000 annually. This is a remote role.",
  "top_k_search": 5,
  "top_k_final": 3
}
```

Response:

```json
{
  "parsed_job_description": {
    "role_title": "Machine Learning Engineer",
    "seniority": "senior",
    "min_experience_years": 4.0
  },
  "total_candidates_retrieved": 5,
  "total_candidates_considered": 5,
  "total_candidates_returned": 3,
  "rankings": [
    {
      "rank": 1,
      "candidate_id": "cand-002",
      "full_name": "Rohan Mehta",
      "final_score": 89.5,
      "match_result": {
        "match_score": 82.5
      },
      "interest_result": {
        "interest_score": 100.0
      }
    },
    {
      "rank": 2,
      "candidate_id": "cand-015",
      "full_name": "Sneha Patel",
      "final_score": 79.0
    },
    {
      "rank": 3,
      "candidate_id": "cand-007",
      "full_name": "Meera Iyer",
      "final_score": 61.0
    }
  ]
}
```
