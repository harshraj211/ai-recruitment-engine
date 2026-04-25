# Project Context

## Goal

Harden the original hackathon talent scouting prototype into a production-oriented talent intelligence engine while keeping the existing FastAPI service boundary, modular service layout, local dataset flow, and demo-friendly frontend.

## Current Architecture

```text
Frontend (static single-page app)
  -> Data Source Selector (Local Dataset / Upload JSON / Simulated External API)
  -> Input Mode Selector (Manual / Demo Data / AI Generate)
  -> POST /api/v1/data-source/local
  -> POST /api/v1/data-source/upload
  -> POST /api/v1/data-source/mock-api
  -> POST /api/v1/generate-jd (AI mode)
  -> POST /api/v1/match/stream
  -> progressive progress + candidate events

FastAPI API Layer
  -> async /health
  -> async /data-source
  -> async /mock-candidates
  -> async /match
  -> async /match/stream (SSE)
  -> async /generate-jd (Groq LLM with fallback)

Pipeline Orchestrator
  -> spaCy JD parsing
  -> Hybrid retrieval (BM25 + dual dense ANN + RRF)
  -> Cross-encoder re-ranking
  -> Weighted skill / experience / role scoring
  -> Predictive engagement scoring
  -> Simulated outreach conversation
  -> Response validation and auto-correction
  -> LLM summary + outreach with PII masking
  -> Final ranking + pagination

Data / Models
  -> Candidate JSON dataset
  -> Runtime candidate source switcher
  -> SentenceTransformers embeddings
  -> FAISS HNSW or IVFFlat indexes
  -> BM25 sparse index
  -> Cross-encoder deep matcher
```

## Major Upgrade Summary

- JD parsing now uses a spaCy pipeline with role, skill, and domain phrase matching.
- Parsed JDs now expose:
  - `mandatory_skills`
  - `nice_to_have_skills`
  - `domain_knowledge`
  - compatibility fields `core_skills` and `secondary_skills`
- Candidate schema now supports `current_company` and `role_history`.
- Skill matching is adjacency-aware through `SkillGraphService`.
- Skill evidence is time-decayed using candidate role history recency.
- Skill coverage is normalized against JD-required skills only, so 7 matched out of 8 required skills scores near 0.875 rather than being diluted by unrelated evidence.
- Retrieval is now hybrid:
  - BM25 sparse search over titles, skills, and companies
  - separate dense profile and skill embeddings
  - FAISS ANN indexes
  - RRF fusion
  - mandatory-skill and salary prefilters
- L2 ranking now includes a cross-encoder stage.
- Interest scoring no longer uses fake recruiter-candidate simulation.
- Predictive engagement now uses deterministic signals:
  - salary alignment
  - availability
  - engagement probability
- Flight risk remains separate and is driven by tenure and career-movement signals.
- LLM usage is restricted to:
  - recruiter outreach generation
  - concise recruiter summaries
  - AI-powered job description generation (via `/generate-jd`)
- Simulated conversational engagement now produces:
  - recruiter/candidate transcript turns for consent, interest, salary, and availability
  - structured signals for consent, interest level, sentiment, confidence, salary alignment, and availability
  - JSON conversation logs under `data/conversations/`
- Outreach prompts reference only the target JD role title, not the candidate's current role title.
- Summary and outreach prompts include verified matched skills, verified missing skills, role, and salary alignment, plus contradiction guards against hallucinated gaps.
- PII is masked before LLM prompts.
- Final ranking is deterministic and favors candidates who satisfy mandatory skills before tie-breaking on score.
- A response validation service now auto-corrects:
  - flat vs nested score mismatches
  - skill chip vs explanation mismatches
  - incorrect missing skill lists
  - contradictory LLM summary/outreach text
- API errors are now returned as structured stage-specific payloads.
- Health checks report candidate data, vector index, and LLM mode.
- Candidate data source mode now supports:
  - Local Dataset using `data/candidates/candidates.json`
  - Upload JSON with validation for `name`, `skills`, `experience`, and `salary`
  - Simulated External API via `GET /api/v1/mock-candidates`
- Data source switches clear candidate/JD parser caches and remove stale FAISS files so the existing matching pipeline uses the active dataset on the next run.
- API responses now include flattened product-facing fields:
  - `candidate_name`
  - `match_score`
  - `interest_score`
  - `final_score`
  - `bm25_score`
  - `cross_encoder_score`
  - `flight_risk_score`
  - `summary`
  - `missing_skills`
  - `recommendation`
  - `engagement_conversation`
- Pagination is supported.
- Frontend now streams progress and candidate cards progressively via SSE.
- Frontend score labels now distinguish `Final Score (Combined)`, `Technical Match Score`, `Interest Score`, and `Re-ranker Score`.
- Frontend includes loading skeletons, structured error rendering, and a deterministic demo fallback shortlist if the live API fails.
- Frontend now includes an Input Mode Selector with three modes:
  - **Manual Input**: free-form textarea (default, unchanged behavior).
  - **Use Demo Data**: dropdown with 5 predefined roles (ML Engineer, Backend Engineer, Data Scientist, Frontend Engineer, DevOps Engineer) that auto-fill the textarea.
  - **Generate with AI**: enter any role title and generate a realistic JD via Groq LLM. Includes loading spinner, success/error status banner, and automatic fallback to demo data if Groq fails.
- Frontend now includes a Data Source Selector with three modes:
  - **Local Dataset**: default active source.
  - **Upload JSON**: accepts a browser JSON file upload and activates validated records.
  - **Simulated External API**: switches to the mock external candidate endpoint.
- Candidate cards now expose the simulated engagement transcript inside the explanation panel so judges can see how interest was assessed.

## Key Services

- `app/services/jd_parser.py`
- `app/services/skill_graph.py`
- `app/services/experience_intelligence.py`
- `app/services/vector_store.py`
- `app/services/cross_encoder_service.py`
- `app/services/match_scoring.py`
- `app/services/interest_scoring.py`
- `app/services/conversation_service.py`
- `app/services/final_ranking.py`
- `app/services/response_validation.py`
- `app/services/ranking_consistency.py`
- `app/services/pipeline_service.py`
- `app/services/candidate_store.py`

## Key Routes

- `app/api/routes/system.py`
- `app/api/routes/data_sources.py`
- `app/api/routes/matching.py`
- `app/api/routes/generate_jd.py`

## API Surface

- `GET /api`
- `GET /api/v1/health`
- `GET /api/v1/data-source`
- `POST /api/v1/data-source/local`
- `POST /api/v1/data-source/upload`
- `GET /api/v1/mock-candidates`
- `POST /api/v1/data-source/mock-api`
- `POST /api/v1/match`
- `POST /api/v1/match/stream`
- `POST /api/v1/generate-jd`

## Current Response Shape Highlights

- `parsed_job_description`
- `rankings[]`
  - `candidate_name`
  - `match_score`
  - `interest_score`
  - `final_score`
  - `bm25_score`
  - `cross_encoder_score`
  - `flight_risk_score`
  - `summary`
  - `missing_skills`
  - `recommendation`
  - nested `match_result`
  - nested `interest_result`

Structured error payloads now use:

- `status`
- `error.code`
- `error.stage`
- `error.message`
- `detail`

## Verified Commands

- `.\\.venv\\Scripts\\python -m pytest tests\\test_data_sources.py -q`
- `.\\.venv\\Scripts\\python -m pytest tests\\test_system_routes.py -q`
- `.\\.venv\\Scripts\\python -m pytest tests\\test_conversation_service.py -q`
- `.\\.venv\\Scripts\\python -m pytest tests -q`
- `.\\.venv\\Scripts\\python scripts\\live_api_test.py`

## Current Test Status

- `43 passed`

## Run Locally

```powershell
.\\.venv\\Scripts\\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Frontend: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

## Notes

- `requirements.txt` now includes `spacy` and `rank-bm25`.
- The local frontend is intentionally static and backend-served for demo simplicity.
- Uploaded candidates are runtime-scoped and do not overwrite `data/candidates/candidates.json`.
- The active source switch lives in `candidate_store`, so the matching pipeline does not need source-specific branches.
- External LLM failures no longer silently change candidate scoring behavior; deterministic scoring and response validation remain the source of truth.
- The `/generate-jd` endpoint uses a dedicated Groq prompt with temperature 0.4 for creative-but-grounded output, and always falls back to a deterministic template on any failure (no API key, network error, short response, etc.).
- The Input Mode Selector only extends input handling; no core pipeline logic was modified.
- The Data Source Selector only changes candidate ingestion and cache invalidation; no core matching logic was modified.
- Conversational engagement is deterministic and simulated by design, matching the hackathon scope without requiring live candidate messaging or authentication.
