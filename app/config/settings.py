from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "Multi-Agent RAG Intelligence Platform"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Ollama / LLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2:latest"
    EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    LLM_TEMPERATURE: float = 0.2
    LLM_TIMEOUT_SECONDS: float = 60.0

    # Qdrant / Vector DB
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "documents_collection"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/app.db"

    # Security
    JWT_SECRET: str = "supersecretjwtkey_change_in_production_32chars"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # RAG parameters
    TOP_K: int = 10
    RERANK_TOP_K: int = 5
    MAX_RETRIES: int = 2
    RETRIEVAL_CONFIDENCE_THRESHOLD: float = 0.65

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
