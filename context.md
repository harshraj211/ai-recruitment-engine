# Project Context

## Goal

Build a hackathon-ready local prototype for an AI-powered talent scouting and engagement agent, following the SRS as the primary source of truth.

## Constraints

- Keep the implementation simple, modular, and executable locally.
- Avoid unnecessary abstractions and multi-agent complexity.
- Use FastAPI, SentenceTransformers (`all-MiniLM-L6-v2`), FAISS, and Groq for conversation later.
- Use Python `3.11` for the real embedding and FAISS flow on Windows.

## Current Status

- Step 1 complete: FastAPI project scaffold and health endpoints.
- Step 2 complete: Local dummy candidate dataset with 20 validated profiles.
- Step 3 complete: Rule-based JD parser for role, skills, experience, salary range, and work mode.
- Step 4 complete: SentenceTransformers embeddings and FAISS-based semantic search.
- Step 5 complete: SRS-style weighted match scoring with explanations.
- Step 6 complete: Recruiter-candidate conversation simulation with transcript storage.
- Step 7 complete: Interest scoring from conversation signals.
- Step 8 complete: Final ranking by combining Match Score and Interest Score.
- Step 9 complete: API endpoint wiring for `/api/v1/match`.

## Latest Implementation Notes

- Match score formula follows the SRS:
  `((0.7 * skill_match) + (0.3 * experience_match)) * 100`
- `skill_match` is the exact overlap ratio between parsed JD skills and candidate skills.
- `experience_match` is `candidate_experience / required_experience`, capped at `1.0`.
- Match explanations include matched skills, missing skills, experience comparison, and the final weighted score.
- The conversation service uses Groq chat completions with JSON object mode when `GROQ_API_KEY` is configured.
- A live `GROQ_API_KEY` is now configured through `.env`, and the Groq conversation path has been verified.
- Conversation transcripts are stored under `data/conversations/` as JSON files.
- Interest scoring uses a transparent weighted formula:
  `3*sentiment + 2*confidence + 2*specificity + 2*salary_match + 1*availability`, normalized to `0-100`.
- Final ranking uses:
  `FinalScore = 0.6 * MatchScore + 0.4 * InterestScore`
- Final ranking now evaluates interest for all semantically retrieved candidates before applying the final shortlist cutoff.
- Each ranked candidate now includes three structured explanation fields:
  `skill_match_reason`, `experience_match_reason`, `conversation_insight`
- The API endpoint is:
  `POST /api/v1/match`

## Key Files

- `app/main.py`
- `app/services/jd_parser.py`
- `app/services/embedding_service.py`
- `app/services/vector_store.py`
- `app/services/match_scoring.py`
- `app/services/conversation_service.py`
- `app/services/final_ranking.py`
- `app/services/interest_scoring.py`
- `app/services/pipeline_service.py`
- `app/api/routes/matching.py`
- `data/candidates/candidates.json`
- `data/conversations/`
- `scripts/preview_match_scoring.py`
- `scripts/preview_conversation.py`
- `scripts/preview_interest_scoring.py`
- `scripts/preview_final_ranking.py`
- `scripts/live_api_test.py`
- `tests/test_match_api.py`
- `README.md`

## Verified Commands

- `python -m pytest -q`
- `py -3.11 -m pytest -q`
- `py -3.11 scripts/build_faiss_index.py`
- `py -3.11 scripts/preview_semantic_search.py`
- `python scripts/preview_match_scoring.py`
- `python scripts/preview_conversation.py`
- `python scripts/preview_interest_scoring.py`
- `py -3.11 scripts/preview_final_ranking.py`
- `py -3.11` end-to-end `/api/v1/match` verification
- `py -3.11` live Groq conversation verification
- `POST /api/v1/match`

## Latest Verified Output

- Top match for the sample JD is `cand-002` (`Rohan Mehta`) with `match_score=82.5`, `interest_score=100.0`, `final_score=89.5`.
- Breakdown: `skill_match=0.75`, `experience_match=1.0`.
- Live Groq conversation verified with provider `groq` and model `llama-3.1-8b-instant`.
- Full live pipeline test via `scripts/live_api_test.py` → HTTP `200`:
  `retrieved=5`, `considered=5`, `returned=3`
  Rank 1: `cand-002` (Rohan Mehta) — final_score 89.5, salary aligned
  Rank 2: `cand-015` (Sneha Patel) — final_score 79.0, salary aligned
  Rank 3: `cand-007` (Meera Iyer) — final_score 69.0, salary above_range
- Each candidate includes structured explanation fields:
  `skill_match_reason`, `experience_match_reason`, `conversation_insight`
- Current test status:
  `py -3.11 -m pytest -q` → `28 passed`

## Status

- System is **demo-ready**. All pipeline stages validated end-to-end with live Groq LLM.
- Start with: `py -3.11 -m uvicorn app.main:app --host 0.0.0.0 --port 8811`
- Test with: `py -3.11 scripts/live_api_test.py`
