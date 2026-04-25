import json
from pathlib import Path

from app.core.config import get_settings
from app.schemas.job_description import ParsedJobDescription
from app.schemas.semantic_search import SemanticSearchResult
from app.services.candidate_store import build_candidate_search_text, load_candidates
from app.services.embedding_service import EmbeddingService


def get_faiss_module():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is not installed. Use Python 3.11 and run `pip install -r requirements.txt`."
        ) from exc

    return faiss


def build_job_search_text(parsed_jd: ParsedJobDescription) -> str:
    parts = []

    if parsed_jd.role_title:
        parts.append(f"Target role: {parsed_jd.role_title}")
    if parsed_jd.seniority:
        parts.append(f"Seniority: {parsed_jd.seniority}")
    if parsed_jd.min_experience_years is not None:
        parts.append(f"Minimum experience: {parsed_jd.min_experience_years} years")
    if parsed_jd.skills:
        parts.append(f"Required skills: {', '.join(parsed_jd.skills)}")
    if parsed_jd.work_mode:
        parts.append(f"Work mode: {parsed_jd.work_mode}")
    if parsed_jd.salary_range_usd:
        parts.append(
            f"Salary range: {parsed_jd.salary_range_usd[0]} to {parsed_jd.salary_range_usd[1]} USD"
        )

    parts.append(f"Original job description: {parsed_jd.raw_text}")
    return ". ".join(parts)


class CandidateVectorStore:
    """Builds and queries a local FAISS index for candidate semantic search."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        index_path: str | None = None,
    ) -> None:
        settings = get_settings()
        self.embedding_service = embedding_service or EmbeddingService()
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.metadata_path = self.index_path.with_suffix(".meta.json")
        self._index = None
        self._metadata: list[dict] = []

    def build_index(self) -> dict[str, int | str]:
        candidates = load_candidates()
        search_texts = [build_candidate_search_text(candidate) for candidate in candidates]
        embeddings = self.embedding_service.embed_texts(search_texts)

        faiss = get_faiss_module()
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))

        metadata = []
        for candidate, search_text in zip(candidates, search_texts, strict=True):
            metadata.append(
                {
                    "candidate_id": candidate.id,
                    "full_name": candidate.full_name,
                    "role_title": candidate.role_title,
                    "total_experience_years": candidate.total_experience_years,
                    "skills": candidate.skills,
                    "profile_summary": candidate.profile_summary,
                    "search_text": search_text,
                }
            )

        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        self._index = index
        self._metadata = metadata

        return {
            "candidate_count": len(metadata),
            "embedding_dimension": int(embeddings.shape[1]),
            "index_path": str(self.index_path),
        }

    def _load_index_if_needed(self) -> None:
        if self._index is not None and self._metadata:
            return

        if not self.index_path.exists() or not self.metadata_path.exists():
            self.build_index()
            return

        faiss = get_faiss_module()
        self._index = faiss.read_index(str(self.index_path))
        self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def search(self, query_text: str, top_k: int = 5) -> list[SemanticSearchResult]:
        self._load_index_if_needed()

        if not self._metadata:
            return []

        query_vector = self.embedding_service.embed_texts([query_text])
        top_k = min(top_k, len(self._metadata))
        scores, indices = self._index.search(query_vector, top_k)

        results = []
        for score, index_position in zip(scores[0], indices[0], strict=True):
            if index_position < 0:
                continue

            record = self._metadata[index_position]
            results.append(
                SemanticSearchResult(
                    candidate_id=record["candidate_id"],
                    full_name=record["full_name"],
                    role_title=record["role_title"],
                    total_experience_years=record["total_experience_years"],
                    skills=record["skills"],
                    similarity_score=float(score),
                    profile_summary=record["profile_summary"],
                )
            )

        return results

    def search_parsed_job(
        self,
        parsed_jd: ParsedJobDescription,
        top_k: int = 5,
    ) -> list[SemanticSearchResult]:
        return self.search(build_job_search_text(parsed_jd), top_k=top_k)
