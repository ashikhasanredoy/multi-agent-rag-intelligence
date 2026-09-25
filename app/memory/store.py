import os
import json
import math
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from app.memory.models import MemoryItem, MemoryCategory, MemoryRecallResult
from app.services.embeddings import embedding_service
from app.utils.logging import logger


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class LongTermMemoryStore:
    """
    Persistent Long-Term Memory (LTM) Store.
    Features:
    - Semantic Vector Search with cosine similarity
    - Lexical Keyword Search for exact entity/tag matching
    - Hybrid Reciprocal Scoring & Category Filtering
    - Automatic JSON persistence in ./data/long_term_memory.json
    """

    def __init__(self, storage_path: str = "./data/long_term_memory.json"):
        self.storage_path = Path(storage_path)
        self.memories: Dict[str, MemoryItem] = {}
        self._lock = asyncio.Lock()
        self._load_from_disk()

    def _load_from_disk(self):
        try:
            if self.storage_path.exists():
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item_data in data:
                        item = MemoryItem(**item_data)
                        self.memories[item.id] = item
                logger.info(f"Loaded {len(self.memories)} long-term memories from {self.storage_path}")
            else:
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_to_disk()
        except Exception as e:
            logger.error(f"Error loading long-term memories: {e}")
            self.memories = {}

    def _save_to_disk(self):
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = [m.model_dump() for m in self.memories.values()]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error persisting long-term memories: {e}")

    async def add_memory(
        self,
        content: str,
        category: MemoryCategory = MemoryCategory.GENERAL,
        importance: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
        embed: bool = True
    ) -> MemoryItem:
        """Add and embed a new memory item."""
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Memory content cannot be empty.")

        # Check for near duplicate memory to avoid redundancy
        for existing in self.memories.values():
            if existing.content.lower() == clean_content.lower() and existing.category == category:
                existing.last_accessed_at = datetime.utcnow().isoformat()
                existing.access_count += 1
                if metadata:
                    existing.metadata.update(metadata)
                self._save_to_disk()
                return existing

        embedding = None
        if embed:
            try:
                embedding = await embedding_service.get_embedding(clean_content)
            except Exception as e:
                logger.warning(f"Could not generate embedding for memory '{clean_content[:30]}...': {e}")

        item = MemoryItem(
            content=clean_content,
            category=category,
            importance=max(1, min(5, importance)),
            embedding=embedding,
            metadata=metadata or {}
        )

        async with self._lock:
            self.memories[item.id] = item
            self._save_to_disk()

        logger.info(f"Added new memory [{item.id}] ({category.value}): '{clean_content[:50]}...'")
        return item

    async def recall_memories(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[MemoryCategory] = None,
        min_score: float = 0.30
    ) -> List[MemoryRecallResult]:
        """
        Recall the most relevant memories for a user query using hybrid semantic + keyword scoring.
        """
        if not self.memories or not query.strip():
            return []

        query_tokens = set(query.lower().split())
        query_embedding = None

        try:
            query_embedding = await embedding_service.get_embedding(query)
        except Exception as e:
            logger.warning(f"Embedding failed during memory recall: {e}")

        scored_memories: List[Tuple[MemoryItem, float]] = []

        for item in self.memories.values():
            # Category filter if specified
            if category and item.category != category:
                continue

            # 1. Semantic Similarity Score (0.0 to 1.0)
            sem_score = 0.0
            if query_embedding and item.embedding:
                sem_score = max(0.0, cosine_similarity(query_embedding, item.embedding))

            # 2. Keyword Overlap Score (0.0 to 1.0)
            item_tokens = set(item.content.lower().split())
            overlap = len(query_tokens.intersection(item_tokens))
            kw_score = min(1.0, overlap / max(1, len(query_tokens)))

            # 3. Importance & Recency boost
            importance_boost = (item.importance - 1) * 0.05  # up to +0.20

            # Combined Hybrid Score
            if query_embedding and item.embedding:
                final_score = (0.7 * sem_score) + (0.3 * kw_score) + importance_boost
            else:
                final_score = kw_score + importance_boost

            if final_score >= min_score:
                scored_memories.append((item, final_score))

        # Sort descending by score
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        results = [
            MemoryRecallResult(memory=mem, relevance_score=round(score, 4))
            for mem, score in scored_memories[:top_k]
        ]

        # Update access stats asynchronously
        for res in results:
            res.memory.last_accessed_at = datetime.utcnow().isoformat()
            res.memory.access_count += 1
        if results:
            self._save_to_disk()

        return results

    def list_memories(self, category: Optional[MemoryCategory] = None) -> List[MemoryItem]:
        """List all memories sorted by creation date descending."""
        items = list(self.memories.values())
        if category:
            items = [m for m in items if m.category == category]
        items.sort(key=lambda x: x.created_at, reverse=True)
        return items

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory item by ID."""
        async with self._lock:
            if memory_id in self.memories:
                del self.memories[memory_id]
                self._save_to_disk()
                logger.info(f"Deleted memory: {memory_id}")
                return True
            return False

    async def clear_memories(self) -> int:
        """Clear all stored memories."""
        async with self._lock:
            count = len(self.memories)
            self.memories.clear()
            self._save_to_disk()
            logger.info(f"Cleared all {count} memories.")
            return count


# Singleton memory store
memory_store = LongTermMemoryStore()
