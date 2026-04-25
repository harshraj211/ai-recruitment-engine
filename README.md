# AI-Powered Talent Scouting & Engagement Agent

A FastAPI-based recruitment intelligence demo that parses job descriptions, retrieves relevant candidates, ranks the shortlist, and generates recruiter-ready explanations. The app is designed for local demos while showing how production systems separate matching logic from candidate data ingestion.

## Highlights

- Static frontend served by FastAPI
- Local candidate dataset with runtime data-source switching
- Upload JSON flow with candidate schema validation
- Simulated external candidate API for demo ingestion
- Job-description parsing with role, skill, salary, and work-mode extraction
- Hybrid retrieval with BM25, dense embeddings, FAISS, and reciprocal rank fusion
- Cross-encoder re-ranking
- Technical match, engagement, flight-risk, and final ranking scores
- SSE streaming for progressive frontend results
- Structured stage-specific API errors
- Deterministic fallback behavior when optional LLM generation is unavailable

## Architecture

```text
Frontend
  -> Data Source selector
  -> Input Mode selector
  -> POST /api/v1/match/stream

FastAPI
  -> /api/v1/data-source
  -> /api/v1/mock-candidates
  -> /api/v1/match
  -> /api/v1/match/stream
  -> /api/v1/generate-jd

Pipeline
  -> JD Parser
  -> Hybrid Retrieval
  -> Cross-Encoder Re-ranker
  -> Match Scoring
  -> Predictive Engagement
  -> Response Validation
  -> Summary / Outreach
  -> Final Ranking + Pagination
```

## Setup

Use Python 3.11 for the full local stack.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Frontend: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/api/v1/health`

Optional `.env` values:

```env
GROQ_API_KEY=
LOG_LEVEL=INFO
```

When `GROQ_API_KEY` is not configured, scoring still runs deterministically and JD generation uses a local fallback template.

## Data Source Modes

The frontend includes a `Data Source` selector:

- `Local Dataset`: uses `data/candidates/candidates.json`.
- `Upload JSON`: validates uploaded records and activates them for the current runtime session.
- `Simulated External API`: loads candidates through a mock external-source endpoint.

Upload records must include:

```json
[
  {
    "name": "Neha Kapoor",
    "skills": ["Python", "FastAPI", "PostgreSQL"],
    "experience": 4,
    "salary": 70000
  }
]
```

The upload flow also accepts `full_name`, `total_experience_years`, and `expected_salary_usd` aliases so full candidate records can be reused.

## API Reference

### `GET /api/v1/data-source`

Returns the active candidate source and candidate count.

### `POST /api/v1/data-source/local`

Switches matching back to the checked-in local dataset.

### `POST /api/v1/data-source/upload`

Activates uploaded candidate JSON after validation.

```json
{
  "candidates": [
    {
      "name": "Arjun Rao",
      "role": "ML Engineer",
      "skills": "Python, PyTorch, AWS",
      "experience": "6",
      "salary": "95k"
    }
  ]
}
```

### `GET /api/v1/mock-candidates`

Returns candidate data in the same structure a real external system could provide.

### `POST /api/v1/data-source/mock-api`

Activates the simulated external API dataset for matching.

### `POST /api/v1/match`

Runs the matching pipeline and returns ranked candidates.

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

### `POST /api/v1/match/stream`

Returns `text/event-stream` events:

- `progress`
- `candidate`
- `result`
- `error`

## Development

Run the test suite:

```powershell
.\.venv\Scripts\python -m pytest tests -q
```

Build or refresh the FAISS index manually:

```powershell
.\.venv\Scripts\python scripts\build_faiss_index.py
```

Runtime artifacts are written under `data/faiss/` and `data/conversations/`, both ignored by git.
