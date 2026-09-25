import httpx
from typing import List, Dict, Any, Optional
from app.config.settings import get_settings
from app.utils.logging import logger


class EmbeddingService:
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.OLLAMA_BASE_URL.rstrip("/")
        self.model = self.settings.EMBEDDING_MODEL

    async def get_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for a single text chunk."""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("embedding", [])
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise RuntimeError(f"Failed to generate embedding: {str(e)}") from e

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of texts sequentially/batched."""
        embeddings = []
        for text in texts:
            emb = await self.get_embedding(text)
            embeddings.append(emb)
        return embeddings


embedding_service = EmbeddingService()
