from pathlib import Path

import pytest

from app.schemas.job_description import ParsedJobDescription
from app.services.vector_store import CandidateVectorStore, build_job_search_text

faiss = pytest.importorskip("faiss")


class FakeEmbeddingService:
    def embed_texts(self, texts: list[str]):
        import numpy as np

        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float(lowered.count("python")),
                    float(lowered.count("machine learning") + lowered.count("ml")),
                    float(lowered.count("react")),
                    float(lowered.count("aws")),
                    float(lowered.count("nlp")),
                ]
            )

        array = np.array(vectors, dtype="float32")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return array / norms


def test_build_job_search_text_includes_taxonomy_fields() -> None:
    parsed_jd = ParsedJobDescription(
        raw_text="Need a remote ML engineer",
        role_title="Machine Learning Engineer",
        seniority="senior",
        min_experience_years=4,
        skills=["Python", "AWS", "Machine Learning"],
        mandatory_skills=["Python", "AWS"],
        nice_to_have_skills=["Machine Learning"],
        salary_range_usd=[50000, 65000],
        work_mode="remote",
    )

    search_text = build_job_search_text(parsed_jd)

    assert "Target role: Machine Learning Engineer" in search_text
    assert "Mandatory skills: Python, AWS" in search_text
    assert "Nice to have skills: Machine Learning" in search_text
    assert "Salary range: 50000 to 65000 USD" in search_text


def test_candidate_vector_store_builds_and_searches(tmp_path: Path) -> None:
    store = CandidateVectorStore(
        embedding_service=FakeEmbeddingService(),
        index_path=str(tmp_path / "candidates.index"),
    )

    summary = store.build_index()
    results = store.search(
        "Looking for a machine learning engineer with python and aws experience",
        top_k=3,
    )

    assert summary["candidate_count"] == 20
    assert (tmp_path / "candidates.index").exists()
    assert (tmp_path / "candidates.profile.index").exists()
    assert (tmp_path / "candidates.skills.index").exists()
    assert (tmp_path / "candidates.meta.json").exists()
    assert len(results) == 3
    assert results[0].similarity_score >= results[1].similarity_score
    assert results[0].bm25_score is not None
