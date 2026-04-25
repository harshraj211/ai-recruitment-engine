import asyncio
import json
import logging
import re
from pathlib import Path
from threading import RLock

from app.core.config import get_settings
from app.schemas.job_description import ParsedJobDescription
from app.schemas.semantic_search import SemanticSearchResult
from app.services.candidate_store import (
    build_candidate_search_text,
    build_candidate_skill_text,
    load_candidates,
)
from app.services.embedding_service import EmbeddingService
from app.services.skill_graph import SkillGraphService

logger = logging.getLogger(__name__)


def get_faiss_module():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is not installed. Use Python 3.11 and run `pip install -r requirements.txt`."
        ) from exc
    return faiss


def get_bm25_class():
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError(
            "rank-bm25 is not installed. Use Python 3.11 and run `pip install -r requirements.txt`."
        ) from exc
    return BM25Okapi


def tokenize_text(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9+]+", text.lower()) if len(token) > 1]


def min_max_normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high <= low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def build_job_search_text(parsed_jd: ParsedJobDescription) -> str:
    parts = []
    if parsed_jd.role_title:
        parts.append(f"Target role: {parsed_jd.role_title}")
    if parsed_jd.seniority:
        parts.append(f"Seniority: {parsed_jd.seniority}")
    if parsed_jd.min_experience_years is not None:
        parts.append(f"Minimum experience: {parsed_jd.min_experience_years} years")
    if parsed_jd.mandatory_skills:
        parts.append(f"Mandatory skills: {', '.join(parsed_jd.mandatory_skills)}")
    elif parsed_jd.skills:
        parts.append(f"Required skills: {', '.join(parsed_jd.skills)}")
    if parsed_jd.nice_to_have_skills:
        parts.append(f"Nice to have skills: {', '.join(parsed_jd.nice_to_have_skills)}")
    if parsed_jd.domain_knowledge:
        parts.append(f"Domain knowledge: {', '.join(parsed_jd.domain_knowledge)}")
    if parsed_jd.work_mode:
        parts.append(f"Work mode: {parsed_jd.work_mode}")
    if parsed_jd.salary_range_usd:
        parts.append(
            f"Salary range: {parsed_jd.salary_range_usd[0]} to {parsed_jd.salary_range_usd[1]} USD"
        )
    parts.append(f"Original job description: {parsed_jd.raw_text}")
    return ". ".join(parts)


def build_job_skill_text(parsed_jd: ParsedJobDescription) -> str:
    parts = [
        f"Mandatory skills: {', '.join(parsed_jd.mandatory_skills)}",
        f"Nice to have skills: {', '.join(parsed_jd.nice_to_have_skills)}",
        f"Domain knowledge: {', '.join(parsed_jd.domain_knowledge)}",
        f"Role: {parsed_jd.role_title or ''}",
    ]
    return ". ".join(part for part in parts if not part.endswith(": "))


class CandidateVectorStore:
    """Hybrid sparse+dense retrieval with BM25, FAISS ANN, and RRF fusion."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        skill_graph_service: SkillGraphService | None = None,
        index_path: str | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.embedding_service = embedding_service or EmbeddingService()
        self.skill_graph_service = skill_graph_service or SkillGraphService()
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.profile_index_path = self.index_path.with_name(f"{self.index_path.stem}.profile.index")
        self.skill_index_path = self.index_path.with_name(f"{self.index_path.stem}.skills.index")
        self.metadata_path = self.index_path.with_suffix(".meta.json")
        self._profile_index = None
        self._skill_index = None
        self._metadata: list[dict] = []
        self._bm25 = None
        self._lock = RLock()

    def _build_ann_index(self, embeddings):
        faiss = get_faiss_module()
        dimension = int(embeddings.shape[1])
        index_type = self.settings.faiss_index_type.lower()

        if index_type == "ivfflat":
            nlist = max(1, min(64, embeddings.shape[0] // 2 or 1))
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            if not index.is_trained:
                index.train(embeddings)
            index.nprobe = self.settings.faiss_nprobe
            index.add(embeddings)
            return index

        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 40
        index.add(embeddings)
        return index

    def build_index(self) -> dict[str, int | str]:
        with self._lock:
            candidates = load_candidates()
            profile_texts = [build_candidate_search_text(candidate) for candidate in candidates]
            skill_texts = [build_candidate_skill_text(candidate) for candidate in candidates]
            profile_embeddings = self.embedding_service.embed_texts(profile_texts)
            skill_embeddings = self.embedding_service.embed_texts(skill_texts)

            profile_index = self._build_ann_index(profile_embeddings)
            skill_index = self._build_ann_index(skill_embeddings)

            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss = get_faiss_module()
            faiss.write_index(profile_index, str(self.profile_index_path))
            faiss.write_index(skill_index, str(self.skill_index_path))
            faiss.write_index(profile_index, str(self.index_path))

            metadata = []
            for candidate, profile_text, skill_text in zip(
                candidates,
                profile_texts,
                skill_texts,
                strict=True,
            ):
                metadata.append(
                    {
                        "candidate_id": candidate.id,
                        "full_name": candidate.full_name,
                        "role_title": candidate.role_title,
                        "total_experience_years": candidate.total_experience_years,
                        "skills": candidate.skills,
                        "profile_summary": candidate.profile_summary,
                        "expected_salary_usd": candidate.expected_salary_usd,
                        "current_company": candidate.current_company,
                        "company_names": candidate.company_names,
                        "profile_text": profile_text,
                        "skill_text": skill_text,
                        "bm25_tokens": tokenize_text(
                            " ".join(
                                [
                                    candidate.role_title,
                                    " ".join(candidate.preferred_roles),
                                    " ".join(candidate.skills),
                                    " ".join(candidate.company_names),
                                ]
                            )
                        ),
                    }
                )

            self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            BM25Okapi = get_bm25_class()
            self._bm25 = BM25Okapi([record["bm25_tokens"] for record in metadata])
            self._profile_index = profile_index
            self._skill_index = skill_index
            self._metadata = metadata

            logger.info("Built hybrid candidate index for %s candidates", len(metadata))
            return {
                "candidate_count": len(metadata),
                "embedding_dimension": int(profile_embeddings.shape[1]),
                "index_type": self.settings.faiss_index_type,
                "index_path": str(self.index_path.parent),
            }

    def _load_index_if_needed(self) -> None:
        with self._lock:
            if self._profile_index is not None and self._skill_index is not None and self._metadata:
                return

            if (
                not self.profile_index_path.exists()
                or not self.skill_index_path.exists()
                or not self.metadata_path.exists()
            ):
                self.build_index()
                return

            faiss = get_faiss_module()
            self._profile_index = faiss.read_index(str(self.profile_index_path))
            self._skill_index = faiss.read_index(str(self.skill_index_path))
            self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if self._metadata and "expected_salary_usd" not in self._metadata[0]:
                self._profile_index = None
                self._skill_index = None
                self._metadata = []
                self.build_index()
                return
            BM25Okapi = get_bm25_class()
            self._bm25 = BM25Okapi([record["bm25_tokens"] for record in self._metadata])

            if self.settings.faiss_index_type.lower() == "ivfflat":
                self._profile_index.nprobe = self.settings.faiss_nprobe
                self._skill_index.nprobe = self.settings.faiss_nprobe

    def _passes_prefilter(self, record: dict, parsed_jd: ParsedJobDescription) -> bool:
        if len(parsed_jd.salary_range_usd) == 2:
            _, high = sorted(parsed_jd.salary_range_usd)
            expected_salary = record.get("expected_salary_usd")
            if expected_salary is not None and expected_salary > high * 1.25:
                return False

        mandatory = parsed_jd.mandatory_skills or parsed_jd.core_skills
        if not mandatory:
            return True

        for required_skill in mandatory:
            _, score = self.skill_graph_service.best_match(required_skill, record["skills"])
            if score < 0.55:
                return False
        return True

    def _prefilter_positions(self, parsed_jd: ParsedJobDescription) -> set[int]:
        positions = set()
        for index, record in enumerate(self._metadata):
            if self._passes_prefilter(record, parsed_jd):
                positions.add(index)
        return positions

    def _dense_scores(
        self,
        query_profile_text: str,
        query_skill_text: str,
        candidate_positions: set[int],
        top_n: int,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        if not candidate_positions:
            return {}, {}, {}

        profile_query, skill_query = self.embedding_service.embed_texts(
            [query_profile_text, query_skill_text]
        )

        dense_profile: dict[str, float] = {}
        dense_skill: dict[str, float] = {}
        search_k = min(max(top_n * 3, 20), len(self._metadata))

        profile_scores, profile_indices = self._profile_index.search(profile_query.reshape(1, -1), search_k)
        skill_scores, skill_indices = self._skill_index.search(skill_query.reshape(1, -1), search_k)

        for score, position in zip(profile_scores[0], profile_indices[0], strict=True):
            if position >= 0 and position in candidate_positions:
                dense_profile[self._metadata[position]["candidate_id"]] = float(score)

        for score, position in zip(skill_scores[0], skill_indices[0], strict=True):
            if position >= 0 and position in candidate_positions:
                dense_skill[self._metadata[position]["candidate_id"]] = float(score)

        dense_profile = min_max_normalize(dense_profile)
        dense_skill = min_max_normalize(dense_skill)
        dense_combined = {}
        for candidate_id in set(dense_profile) | set(dense_skill):
            dense_combined[candidate_id] = (0.65 * dense_profile.get(candidate_id, 0.0)) + (
                0.35 * dense_skill.get(candidate_id, 0.0)
            )
        return dense_profile, dense_skill, dense_combined

    def _bm25_scores(
        self,
        query_text: str,
        candidate_positions: set[int],
    ) -> dict[str, float]:
        if not candidate_positions:
            return {}
        raw_scores = self._bm25.get_scores(tokenize_text(query_text))
        filtered = {
            self._metadata[position]["candidate_id"]: float(raw_scores[position])
            for position in candidate_positions
        }
        return min_max_normalize(filtered)

    def _rrf_scores(
        self,
        bm25_scores: dict[str, float],
        dense_scores: dict[str, float],
    ) -> dict[str, float]:
        rrf_scores: dict[str, float] = {}
        for ranking in (
            sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)[:100],
            sorted(dense_scores.items(), key=lambda item: item[1], reverse=True)[:100],
        ):
            for rank, (candidate_id, _) in enumerate(ranking, start=1):
                rrf_scores[candidate_id] = rrf_scores.get(candidate_id, 0.0) + (
                    1.0 / (self.settings.rrf_k + rank)
                )

        return min_max_normalize(rrf_scores)

    def search(self, query_text: str, top_k: int = 5) -> list[SemanticSearchResult]:
        self._load_index_if_needed()
        if not self._metadata:
            return []

        candidate_positions = set(range(len(self._metadata)))
        bm25_scores = self._bm25_scores(query_text, candidate_positions)
        dense_profile, dense_skill, dense_combined = self._dense_scores(
            query_text,
            query_text,
            candidate_positions,
            top_n=top_k,
        )
        rrf_scores = self._rrf_scores(bm25_scores, dense_combined)
        return self._build_results(rrf_scores, bm25_scores, dense_profile, dense_skill, dense_combined, top_k)

    def _build_results(
        self,
        rrf_scores: dict[str, float],
        bm25_scores: dict[str, float],
        dense_profile_scores: dict[str, float],
        dense_skill_scores: dict[str, float],
        dense_scores: dict[str, float],
        top_k: int,
    ) -> list[SemanticSearchResult]:
        metadata_lookup = {record["candidate_id"]: record for record in self._metadata}
        ranked_candidate_ids = sorted(rrf_scores, key=lambda candidate_id: rrf_scores[candidate_id], reverse=True)

        results = []
        for candidate_id in ranked_candidate_ids[:top_k]:
            record = metadata_lookup[candidate_id]
            results.append(
                SemanticSearchResult(
                    candidate_id=record["candidate_id"],
                    full_name=record["full_name"],
                    role_title=record["role_title"],
                    total_experience_years=record["total_experience_years"],
                    skills=record["skills"],
                    similarity_score=float(rrf_scores.get(candidate_id, 0.0)),
                    semantic_similarity_score=float(dense_scores.get(candidate_id, 0.0)),
                    keyword_match_score=float(bm25_scores.get(candidate_id, 0.0)),
                    bm25_score=float(bm25_scores.get(candidate_id, 0.0)),
                    dense_profile_score=float(dense_profile_scores.get(candidate_id, 0.0)),
                    dense_skill_score=float(dense_skill_scores.get(candidate_id, 0.0)),
                    rrf_score=float(rrf_scores.get(candidate_id, 0.0)),
                    profile_summary=record["profile_summary"],
                )
            )
        return results

    def search_parsed_job(
        self,
        parsed_jd: ParsedJobDescription,
        top_k: int = 5,
    ) -> list[SemanticSearchResult]:
        self._load_index_if_needed()
        if not self._metadata:
            return []

        candidate_positions = self._prefilter_positions(parsed_jd)
        if not candidate_positions:
            logger.info("Hybrid prefilters removed all candidates; falling back to full pool.")
            candidate_positions = set(range(len(self._metadata)))

        query_text = build_job_search_text(parsed_jd)
        skill_query = build_job_skill_text(parsed_jd)
        bm25_scores = self._bm25_scores(query_text, candidate_positions)
        dense_profile, dense_skill, dense_combined = self._dense_scores(
            query_text,
            skill_query,
            candidate_positions,
            top_n=min(max(top_k, 10), len(candidate_positions)),
        )
        rrf_scores = self._rrf_scores(bm25_scores, dense_combined)
        return self._build_results(
            rrf_scores,
            bm25_scores,
            dense_profile,
            dense_skill,
            dense_combined,
            top_k,
        )

    async def search_parsed_job_async(
        self,
        parsed_jd: ParsedJobDescription,
        top_k: int = 5,
    ) -> list[SemanticSearchResult]:
        return await asyncio.to_thread(self.search_parsed_job, parsed_jd, top_k)
