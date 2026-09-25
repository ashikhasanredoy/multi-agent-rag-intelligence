from typing import Generator
from app.config.settings import Settings, get_settings
from app.services.llm import LLMService, llm_service
from app.services.embeddings import EmbeddingService, embedding_service


def get_app_settings() -> Settings:
    return get_settings()


def get_llm_service() -> LLMService:
    return llm_service


def get_embedding_service() -> EmbeddingService:
    return embedding_service
