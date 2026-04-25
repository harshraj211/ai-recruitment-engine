from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI-Powered Talent Scouting & Engagement Agent"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    groq_model: str = "llama-3.1-8b-instant"
    groq_temperature: float = 0.0
    groq_api_key: str = ""
    candidate_data_path: str = "data/candidates/candidates.json"
    faiss_index_path: str = "data/faiss/candidates.index"
    faiss_index_type: str = "hnsw"
    faiss_nprobe: int = 8
    hybrid_semantic_weight: float = 0.6
    hybrid_keyword_weight: float = 0.4
    rrf_k: int = 60
    top_k_rerank: int = 20
    conversation_log_path: str = "data/conversations"
    pipeline_stage_timeout_seconds: float = 120.0
    candidate_scoring_timeout_seconds: float = 10.0
    llm_timeout_seconds: float = 20.0
    llm_max_retries: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
