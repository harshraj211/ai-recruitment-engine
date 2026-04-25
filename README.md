# AI-Powered Talent Scouting & Engagement Agent

Production-oriented upgrade of the original talent scouting prototype. The system now uses hybrid retrieval, deep re-ranking, predictive engagement scoring, explainable outputs, and streaming UX while staying inside the existing FastAPI + local data architecture.

## What Changed

- Async FastAPI pipeline and async orchestration
- spaCy-based JD parsing with skill taxonomy
- `mandatory_skills`, `nice_to_have_skills`, `domain_knowledge`
- Skill adjacency matching through `SkillGraphService`
- Time-decayed skill evidence from candidate role history
- JD-normalized skill coverage scoring based on required skills only
- Hybrid L1 retrieval:
  - BM25 sparse search with `rank-bm25`
  - dual dense embeddings for profile and skill text
  - FAISS ANN indexes (`hnsw` by default, `ivfflat` supported)
  - reciprocal rank fusion
  - mandatory-skill and salary prefilters
- L2 re-ranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Deterministic predictive engagement and flight-risk scoring
- LLM use restricted to recruiter summaries and outreach
- Outreach and summary prompts grounded to target-role and verified skill lists
- PII masking before LLM prompts
- SSE streaming endpoint for progressive frontend rendering
- Pagination-ready API responses
- Strong penalty for candidates missing mandatory skills

## Architecture

```text
Frontend
  -> /api/v1/match/stream

FastAPI
  -> /api/v1/match
  -> /api/v1/match/stream

Pipeline
  -> JD Parser
  -> Hybrid Retrieval
  -> Cross-Encoder Re-ranker
  -> Match Scoring
  -> Predictive Engagement
  -> Summary / Outreach
  -> Final Ranking + Pagination
```

## Core Modules

- `app/services/jd_parser.py`
- `app/services/skill_graph.py`
- `app/services/experience_intelligence.py`
- `app/services/vector_store.py`
- `app/services/cross_encoder_service.py`
- `app/services/match_scoring.py`
- `app/services/interest_scoring.py`
- `app/services/conversation_service.py`
- `app/services/final_ranking.py`
- `app/services/pipeline_service.py`

## Install

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
```

Python 3.11 is still the recommended runtime for the full local stack.

## Run

```powershell
.\\.venv\\Scripts\\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Frontend: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/api/v1/health`

## API

### `POST /api/v1/match`

Request body:

```json
{
  "job_description": "We are hiring a Senior Machine Learning Engineer. Must have Python, FastAPI, PyTorch, Docker, AWS, MLflow, and vector search. Nice to have RAG. Budget is $50,000 to $65,000 and the role is remote.",
  "top_k_search": 10,
  "top_k_final": 5,
  "page": 1,
  "page_size": 5,
  "include_outreach": true
}
```

Response highlights:

```json
{
  "parsed_job_description": {
    "role_title": "Machine Learning Engineer",
    "mandatory_skills": ["Python", "FastAPI", "PyTorch", "Docker", "AWS", "MLflow", "Vector Search"],
    "nice_to_have_skills": ["RAG"]
  },
  "rankings": [
    {
      "candidate_name": "Rohan Mehta",
      "match_score": 88.3,
      "interest_score": 74.1,
      "final_score": 82.7,
      "bm25_score": 0.91,
      "cross_encoder_score": 82.0,
      "flight_risk_score": 68.0,
      "summary": "Strong backend ML engineer with high core-skill alignment and good switch likelihood.",
      "missing_skills": ["Vector Search"],
      "recommendation": "Advance to recruiter screen."
    }
  ],
  "page": 1,
  "total_pages": 1
}
```

Score semantics:

- `final_score`: combined ranking score after technical match, engagement, cross-encoder contribution, and mandatory-skill penalty
- `match_score`: technical match score only
- `cross_encoder_score`: deep re-ranker signal shown separately from the combined score

### `POST /api/v1/match/stream`

Returns `text/event-stream` with:

- `progress`
- `candidate`
- `result`
- `error`

## Verification

```powershell
.\\.venv\\Scripts\\python -m pytest tests -q
.\\.venv\\Scripts\\python scripts\\live_api_test.py
```

Current status: `35 passed`
