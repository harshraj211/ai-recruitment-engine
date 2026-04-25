from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI-Powered Talent Scouting & Engagement Agent"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    groq_model: str = "llama-3.1-8b-instant"
    groq_temperature: float = 0.2
    groq_api_key: str = ""
    candidate_data_path: str = "data/candidates/candidates.json"
    faiss_index_path: str = "data/faiss/candidates.index"
    conversation_log_path: str = "data/conversations"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
