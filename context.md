# Project Context

## Goal

Harden the original hackathon talent scouting prototype into a production-oriented talent intelligence engine while keeping the existing FastAPI service boundary, modular service layout, local dataset flow, and demo-friendly frontend.

## Current Architecture

```text
Frontend (static single-page app)
  -> POST /api/v1/match/stream
  -> progressive progress + candidate events

FastAPI API Layer
  -> async /health
  -> async /match
  -> async /match/stream (SSE)

Pipeline Orchestrator
  -> spaCy JD parsing
  -> Hybrid retrieval (BM25 + dual dense ANN + RRF)
  -> Cross-encoder re-ranking
  -> Weighted skill / experience / role scoring
  -> Predictive engagement scoring
  -> Response validation and auto-correction
  -> LLM summary + outreach with PII masking
  -> Final ranking + pagination

Data / Models
  -> Candidate JSON dataset
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
- Pagination is supported.
- Frontend now streams progress and candidate cards progressively via SSE.
- Frontend score labels now distinguish `Final Score (Combined)`, `Technical Match Score`, `Interest Score`, and `Re-ranker Score`.
- Frontend includes loading skeletons, structured error rendering, and a deterministic demo fallback shortlist if the live API fails.

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

## API Surface

- `GET /api`
- `GET /api/v1/health`
- `POST /api/v1/match`
- `POST /api/v1/match/stream`

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

- `.\\.venv\\Scripts\\python -m pytest tests -q`
- `.\\.venv\\Scripts\\python scripts\\live_api_test.py`

## Current Test Status

- `38 passed`

## Run Locally

```powershell
.\\.venv\\Scripts\\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Frontend: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

## Notes

- `requirements.txt` now includes `spacy` and `rank-bm25`.
- The local frontend is intentionally static and backend-served for demo simplicity.
- External LLM failures no longer silently change candidate scoring behavior; deterministic scoring and response validation remain the source of truth.
