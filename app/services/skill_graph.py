from __future__ import annotations

import re
from functools import lru_cache


ADJACENCY_WEIGHTS: dict[str, dict[str, float]] = {
    "kubernetes": {"docker swarm": 0.70, "docker": 0.45, "terraform": 0.35},
    "docker swarm": {"kubernetes": 0.70, "docker": 0.60},
    "vector search": {
        "rag": 0.78,
        "vector databases": 0.85,
        "retrieval augmented generation": 0.72,
    },
    "vector databases": {"vector search": 0.85, "rag": 0.70},
    "machine learning": {"pytorch": 0.62, "scikit learn": 0.60, "mlflow": 0.38},
    "nlp": {"spacy": 0.70, "sentence transformers": 0.75, "information extraction": 0.72},
    "transformers": {"hugging face": 0.74, "sentence transformers": 0.68},
    "aws": {"terraform": 0.42, "kubernetes": 0.34},
    "fastapi": {"rest apis": 0.64, "python": 0.35},
    "mlflow": {"mlops": 0.70, "monitoring": 0.48},
}


def normalize_skill(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+]+", " ", value.lower())).strip()


def tokenize_skill(value: str) -> set[str]:
    return {token for token in normalize_skill(value).split() if len(token) > 1}


class SkillGraphService:
    """Lightweight skill adjacency graph for fuzzy recruiter-style matching."""

    def __init__(self, adjacency_weights: dict[str, dict[str, float]] | None = None) -> None:
        raw_graph = adjacency_weights or ADJACENCY_WEIGHTS
        self.graph = {
            normalize_skill(source): {
                normalize_skill(target): weight for target, weight in targets.items()
            }
            for source, targets in raw_graph.items()
        }

    @staticmethod
    @lru_cache(maxsize=2048)
    def lexical_similarity(left: str, right: str) -> float:
        left_tokens = tokenize_skill(left)
        right_tokens = tokenize_skill(right)
        if not left_tokens or not right_tokens:
            return 0.0

        overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        normalized_left = normalize_skill(left)
        normalized_right = normalize_skill(right)
        if normalized_left in normalized_right or normalized_right in normalized_left:
            overlap = max(overlap, 0.65)
        return overlap

    def similarity(self, required_skill: str, candidate_skill: str) -> float:
        left = normalize_skill(required_skill)
        right = normalize_skill(candidate_skill)
        if left == right:
            return 1.0

        direct = self.graph.get(left, {}).get(right)
        if direct is not None:
            return direct

        reverse = self.graph.get(right, {}).get(left)
        if reverse is not None:
            return reverse

        lexical = self.lexical_similarity(left, right)
        if lexical >= 0.5:
            return min(0.8, lexical)
        return lexical

    def best_match(
        self,
        required_skill: str,
        candidate_skills: list[str],
    ) -> tuple[str | None, float]:
        best_skill = None
        best_score = 0.0
        for skill in candidate_skills:
            score = self.similarity(required_skill, skill)
            if score > best_score:
                best_skill = skill
                best_score = score
        return best_skill, best_score
